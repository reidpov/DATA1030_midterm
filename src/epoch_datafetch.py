"""
Download the four Epoch AI tables used by build_chess_dataset.py from the Airtable API.

REQUIRED: .env file with AIRTABLE_PERSONAL_ACCESS_TOKEN and AIRTABLE_BASE_ID.
"""
import os

import pandas as pd
from dotenv import load_dotenv
from pyairtable import Api

load_dotenv()
api = Api(os.environ["AIRTABLE_PERSONAL_ACCESS_TOKEN"])
base_id = os.environ["AIRTABLE_BASE_ID"]

TABLES = {
    "benchmarks/tasks": "data/raw_tasks.csv",
    "benchmarks/runs": "data/raw_runs.csv",
    "Model versions": "data/raw_model_versions.csv",
    "ML Models": "data/raw_ml_models.csv",
}

for table_name, path in TABLES.items():
    records = api.table(base_id, table_name).all()
    df = pd.DataFrame([r["fields"] for r in records], index=[r["id"] for r in records])
    df.to_csv(path, index_label="record_id")
    print(f"{table_name}: {len(df)} rows -> {path}")