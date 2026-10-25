# Predicting LLM Performance on Chess Puzzles using Public Data

Sam Infante, Noah Kim, Jaeho Lee, Reid Povinelli
DATA1030 (Fall 2026), Brown University

We predict a model version's score on Epoch AI's Chess Puzzles v2.0.0 benchmark from public model information (release date, developer, reasoning level, pricing, and Artificial Analysis benchmark scores), using data pulled from Epoch AI and OpenRouter.

## Structure
- data/ : raw Epoch and OpenRouter data, merged dataset, and cleaned dataset
- src/ : data download (epoch_datafetch.py), dataset builder (build_chess_dataset.py), and EDA (eda.ipynb)
- figures/ : report figures
- report/ : midterm report

## Usage
Python 3.14.6; packages listed in environment.yml
Run 'python src/build_chess_dataset.py', then 'src/eda.ipynb'

## License
MIT

