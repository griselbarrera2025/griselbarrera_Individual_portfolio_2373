# NewsBot Intelligence 2.0

## Overview
End-to-end news pipeline: preprocess → features (TF-IDF) → classifier → metrics.

## How to Run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python classifier.py --train sample_data/train.csv --test sample_data/test.csv --out outputs/results.json
```

## Data
Put CSVs in `sample_data/` with columns: `text,label`.
