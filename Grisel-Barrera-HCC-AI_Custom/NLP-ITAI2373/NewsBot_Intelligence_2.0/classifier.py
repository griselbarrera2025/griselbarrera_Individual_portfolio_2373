#!/usr/bin/env python3
import argparse, os, json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    # fallback tiny demo
    return pd.DataFrame({
        "text": ["Market rallies on profits", "Team secures playoff spot", "New AI chip announced"],
        "label": ["business", "sport", "tech"]
    })

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="sample_data/train.csv")
    p.add_argument("--test", default="sample_data/test.csv")
    p.add_argument("--out", default="outputs/results.json")
    args = p.parse_args()

    train = load_csv(args.train)
    test = load_csv(args.test)

    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    Xtr = vec.fit_transform(train["text"].astype(str))
    Xte = vec.transform(test["text"].astype(str))

    clf = LogisticRegression(max_iter=300)
    clf.fit(Xtr, train["label"].astype(str))
    preds = clf.predict(Xte)

    acc = accuracy_score(test["label"].astype(str), preds)
    rep = classification_report(test["label"].astype(str), preds, output_dict=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "report": rep}, f, indent=2)
    print(f"Saved {args.out}  •  Accuracy={acc:.3f}")
