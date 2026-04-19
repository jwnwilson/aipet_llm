"""Train sklearn TF-IDF + classifier to predict need||action from simple text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from aipet_distill.jsonl import iter_jsonl
from aipet_distill.simple_format import LABEL_SEP, game_context_to_simple_text
from aipet_distill.validate import validate_turn


def _row_to_label(out: dict) -> str:
    dec = out.get("decision") or {}
    need = dec.get("primary_need") or dec.get("action", {}).get("need")
    act = (dec.get("action") or {}).get("name")
    if not need or not act:
        raise ValueError("missing decision.primary_need or decision.action.name")
    return f"{need}{LABEL_SEP}{act}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="JSONL with context + PetTurn output")
    ap.add_argument("--output", type=Path, default=Path("models/action_classifier.joblib"))
    ap.add_argument("--config", type=Path, default=Path("configs/two_stage.yaml"))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    clf_cfg = cfg.get("classifier", {})

    texts: list[str] = []
    labels: list[str] = []
    for obj in iter_jsonl(args.input):
        ctx = obj.get("context") or obj.get("input")
        out = obj.get("output") or obj.get("target")
        if ctx is None or out is None:
            continue
        if isinstance(out, str):
            out = json.loads(out)
        errs = validate_turn(ctx, out)
        if errs:
            print(f"skip invalid: {errs}")
            continue
        texts.append(game_context_to_simple_text(ctx))
        labels.append(_row_to_label(out))

    if not texts:
        raise SystemExit("No training rows.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    ngram = tuple(clf_cfg.get("char_ngram_range", [2, 5]))
    max_feat = clf_cfg.get("max_features", 8000)

    uniq = set(labels)
    if len(uniq) == 1:
        pipe = Pipeline(
            [
                ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=ngram)),
                ("clf", DummyClassifier(strategy="constant", constant=labels[0])),
            ]
        )
    else:
        pipe = Pipeline(
            [
                ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, max_features=max_feat)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=int(clf_cfg.get("max_iter", 400)),
                        solver=str(clf_cfg.get("solver", "lbfgs")),
                    ),
                ),
            ]
        )

    pipe.fit(texts, labels)
    joblib.dump(pipe, args.output)
    print(f"wrote {args.output} ({len(texts)} rows, {len(uniq)} labels)")


if __name__ == "__main__":
    main()
