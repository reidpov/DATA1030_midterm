"""Build one Chess Puzzles 2.0.0 row per OpenRouter-matched Epoch version.

Uses the four Epoch CSVs and a saved OpenRouter /api/v1/models
snapshot. Run ``python src/build_chess_dataset.py`` from any directory.
Pass ``--refresh-openrouter`` to replace the snapshot with one API request.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CATALOG = DATA / "raw_openrouter_models.json"
OUTPUT = DATA / "chess_v2_model_versions.csv"
OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
REASONING_LEVELS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}


def single_link(value: object) -> str | None:
    """Decode a one-item Airtable relationship, leaving other shapes blank."""
    if not isinstance(value, str):
        return None
    try:
        links = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None
    if isinstance(links, list) and len(links) == 1 and isinstance(links[0], str):
        return links[0]
    return None


def normalize_name(value: object) -> str:
    if not isinstance(value, str):
        return ""
    name = value.split(": ", 1)[-1]
    return re.sub(r"[^a-z0-9]", "", name.lower())


def label_without_qualifier(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return re.sub(r"\s*\([^()]*\)\s*$", "", value).strip()


def flatten(value: object, prefix: str = "") -> dict[str, object]:
    """Flatten all catalog attributes; preserve arrays as JSON strings."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(flatten(item, f"{prefix}__{key}" if prefix else key))
        return result
    if isinstance(value, list):
        return {prefix: json.dumps(value, ensure_ascii=False)}
    return {prefix: value}


def read_catalog(path: Path, refresh: bool) -> list[dict]:
    if refresh:
        request = Request(OPENROUTER_URL, headers={"User-Agent": "DATA1030-chess-dataset/1.0"})
        with urlopen(request, timeout=20) as response:
            payload = json.load(response)
        models = payload["data"]
        if not isinstance(models, list) or not models:
            raise ValueError("OpenRouter returned no model catalog entries")
        path.write_text(json.dumps({
            "source": OPENROUTER_URL,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "models": models,
        }, ensure_ascii=False, indent=2) + "\n")
    else:
        saved = json.loads(path.read_text())
        models = saved["models"] if isinstance(saved, dict) else saved
    if not isinstance(models, list) or not all(isinstance(m, dict) for m in models):
        raise ValueError("Expected the OpenRouter catalog to be a list of model objects")
    return models


def add_prefix(table: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return table.rename(columns={c: f"{prefix}__{c}" for c in table.columns})


def build(catalog: list[dict]) -> pd.DataFrame:
    runs = pd.read_csv(DATA / "raw_runs.csv", low_memory=False)
    versions = pd.read_csv(DATA / "raw_model_versions.csv", low_memory=False)
    models = pd.read_csv(DATA / "raw_ml_models.csv", low_memory=False)
    tasks = pd.read_csv(DATA / "raw_tasks.csv", low_memory=False)

    chess_ids = tasks.loc[tasks["Name"].eq("Chess Puzzles"), "record_id"]
    if len(chess_ids) != 1:
        raise ValueError("Expected exactly one Chess Puzzles task")
    runs["epoch_task_record_id"] = runs["task"].map(single_link)
    runs["epoch_version_record_id"] = runs["model"].map(single_link)
    selected = runs.loc[
        runs["epoch_task_record_id"].eq(chess_ids.iloc[0])
        & runs["task version"].eq("2.0.0")
        & runs["Status"].eq("Success")
    ].copy()
    selected["run_started_utc"] = pd.to_datetime(selected["started_at"], errors="coerce", utc=True)
    selected = selected.sort_values(["run_started_utc", "record_id"], na_position="first")
    selected = selected.drop_duplicates("epoch_version_record_id", keep="last")
    if selected["epoch_version_record_id"].isna().any():
        raise ValueError("A selected run lacks a model-version link")

    versions["epoch_base_model_record_id"] = versions["Model"].map(single_link)
    selected = add_prefix(selected, "run").rename(columns={
        "run__epoch_task_record_id": "epoch_task_record_id",
        "run__epoch_version_record_id": "epoch_version_record_id",
        "run__run_started_utc": "run_started_utc",
    })
    versions = add_prefix(versions, "version").rename(columns={
        "version__record_id": "epoch_version_record_id",
        "version__epoch_base_model_record_id": "epoch_base_model_record_id",
    })
    models = add_prefix(models, "base_model").rename(columns={
        "base_model__record_id": "epoch_base_model_record_id"
    })
    tasks = add_prefix(tasks, "task").rename(columns={
        "task__record_id": "epoch_task_record_id"
    })
    dataset = (
        selected.merge(versions, on="epoch_version_record_id", how="left", validate="many_to_one")
        .merge(models, on="epoch_base_model_record_id", how="left", validate="many_to_one")
        .merge(tasks, on="epoch_task_record_id", how="left", validate="many_to_one")
    )
    dataset.insert(0, "score_target", pd.to_numeric(dataset["run__Best score (across scorers)"], errors="coerce"))
    if dataset["score_target"].isna().any() or not dataset["score_target"].between(0, 1).all():
        raise ValueError("Selected benchmark scores are missing or outside 0–1")

    dataset["model_version_name"] = dataset["version__Unique display name"].combine_first(
        dataset["version__Display name"]
    ).combine_first(dataset["version__id"])
    dataset["base_model_name"] = dataset["base_model__Model"]
    dataset["version_name_without_qualifier"] = dataset["model_version_name"].map(label_without_qualifier)
    dataset["version_name_qualifier"] = dataset["model_version_name"].str.extract(r"\(([^()]*)\)\s*$", expand=False)
    name_level = dataset["version_name_qualifier"].str.lower().replace({"no thinking": "none"})
    dataset["reasoning_level_from_name"] = name_level.where(name_level.isin(REASONING_LEVELS))
    slug_suffix = dataset["version__id"].str.extract(r"_([^_]+)$", expand=False).str.lower()
    dataset["reasoning_level_from_slug"] = slug_suffix.where(slug_suffix.isin(REASONING_LEVELS))
    dataset["reasoning_level"] = dataset["reasoning_level_from_name"].combine_first(
        dataset["reasoning_level_from_slug"]
    )
    dataset["reasoning_mode_from_name"] = dataset["version_name_qualifier"].str.lower().where(
        dataset["version_name_qualifier"].str.lower().isin({"thinking", "no thinking", "unknown thinking"})
    )
    dataset["thinking_budget_label_from_name"] = dataset["version_name_qualifier"].where(
        dataset["version_name_qualifier"].str.fullmatch(r"\d+k thinking", case=False, na=False)
    )

    catalog_rows = []
    for entry in catalog:
        row = flatten(entry, "openrouter")
        row["openrouter__created_utc"] = (
            datetime.fromtimestamp(entry["created"], tz=timezone.utc).isoformat()
            if isinstance(entry.get("created"), (int, float)) else None
        )
        catalog_rows.append(row)
    catalog_table = pd.DataFrame(catalog_rows)
    if catalog_table["openrouter__id"].duplicated().any():
        raise ValueError("Duplicate OpenRouter IDs in catalog")

    by_name: dict[str, list[str]] = {}
    for entry in catalog:
        name_key = normalize_name(entry.get("name"))
        if name_key:
            by_name.setdefault(name_key, []).append(entry["id"])

    def match(row: pd.Series) -> pd.Series:
        specific_variant = isinstance(row["base_model_name"], str) and "hosted" in row["base_model_name"].lower()
        candidates = [
            ("base_model_name_exact", normalize_name(row["base_model_name"]), by_name),
            ("version_name_exact", normalize_name(row["version_name_without_qualifier"])
             if not specific_variant else "", by_name),
        ]
        for method, key, lookup in candidates:
            ids = list(dict.fromkeys(lookup.get(key, []))) if key else []
            if len(ids) == 1:
                return pd.Series({"openrouter_match_method": method, "openrouter_model_id": ids[0]})
        return pd.Series({"openrouter_match_method": "unmatched", "openrouter_model_id": None})

    dataset = pd.concat([dataset, dataset.apply(match, axis=1)], axis=1)
    dataset = dataset.loc[dataset["openrouter_model_id"].notna()].copy()
    dataset = dataset.merge(catalog_table, left_on="openrouter_model_id", right_on="openrouter__id",
                            how="left", validate="many_to_one")
    if not dataset["epoch_version_record_id"].is_unique:
        raise ValueError("Final model-version IDs are not unique")
    return dataset.sort_values("epoch_version_record_id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-openrouter", action="store_true", help="Fetch the catalog once and replace the saved snapshot")
    args = parser.parse_args()
    dataset = build(read_catalog(CATALOG, args.refresh_openrouter))
    dataset.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(dataset)} rows and {len(dataset.columns)} columns to {OUTPUT}")
    print("OpenRouter matches:", dataset["openrouter_match_method"].value_counts().to_dict())


if __name__ == "__main__":
    main()
