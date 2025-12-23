Human vs AI Text Classifier (Sinhala)

What this includes
- `human_ai_classifier.ipynb`: Jupyter notebook with data loading, preprocessing, TF-IDF + LogisticRegression baseline, evaluation, and an optional transformer fine-tuning stub.
- `requirements.txt`: minimal dependencies (transformers/datasets/torch are optional and only needed for transformer tuning).

Quick start (bash)

1. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the notebook (e.g., with `jupyter lab` or `jupyter notebook`):

```bash
jupyter lab
# open human_ai_classifier.ipynb
```

3. Tips
- The notebook uses a lightweight TF-IDF baseline suitable for quick iteration. Transformer fine-tuning is provided as a commented stub and requires GPU for practical training.
- Place `train.jsonl`, `val.jsonl`, `test.jsonl` in the `data/` folder. Each line should be a JSON object with at least `text` and `label` fields.

Backend API (FastAPI)

1) Install deps (or reuse the same env as above):

```bash
pip install -r requirements.txt
```

2) Run the API:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

3) Example requests:

```bash
curl -X POST "http://localhost:8000/classify" \
	-H "Content-Type: application/json" \
	-d '{"text": "සිංහල පෙළ උදාහරණයක්"}'

curl -X POST "http://localhost:8000/classify-batch" \
	-H "Content-Type: application/json" \
	-d '{"texts": ["පෙළ 1", "පෙළ 2"], "return_probabilities": true}'
```

Environment variables
- `MODEL_PATH`: Optional override of the model directory (default: `models/bert_multilingual_model`).

If you want, I can run the notebook's baseline training here (if you allow me to run commands), or add a ready-to-run script (`train.py`) for training from the command line.