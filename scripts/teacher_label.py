#!/usr/bin/env python3
"""Call an OpenAI-compatible chat API to label each context with a PetTurn JSONL row."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from aipet_distill.jsonl import iter_jsonl, write_jsonl
from aipet_distill.prompts import SYSTEM, format_context_json
from aipet_distill.validate import validate_turn


def _post_json(url: str, headers: dict[str, str], payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _context_key(ctx: dict) -> str:
    stable = json.dumps(ctx, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="JSONL with {\"context\": {...}} per line")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--api-url", default=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1/chat/completions"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--cache-file", type=Path, default=Path(".cache/teacher_label_cache.json"))
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--retry-backoff-seconds", type=float, default=1.5)
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("Set OPENAI_API_KEY or pass --api-key")

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"}
    args.cache_file.parent.mkdir(parents=True, exist_ok=True)
    if args.cache_file.exists():
        cache = json.loads(args.cache_file.read_text(encoding="utf-8"))
        if not isinstance(cache, dict):
            cache = {}
    else:
        cache = {}

    out_rows: list[dict] = []
    for obj in iter_jsonl(args.input):
        ctx = obj.get("context") or obj.get("input")
        if ctx is None:
            print("skip: no context")
            continue
        user = (
            "Game context JSON:\n"
            f"{format_context_json(ctx)}\n\n"
            "Reply with a single JSON object matching the PetTurn schema from the system message."
        )
        payload = {
            "model": args.model,
            "temperature": 0.3,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
        }
        cache_key = _context_key(ctx)
        content = cache.get(cache_key)
        try:
            if content is None:
                last_error: Exception | None = None
                for attempt in range(1, args.max_retries + 1):
                    try:
                        raw = _post_json(args.api_url, headers, payload)
                        content = raw["choices"][0]["message"]["content"]
                        cache[cache_key] = content
                        break
                    except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as e:
                        last_error = e
                        if attempt < args.max_retries:
                            sleep_s = args.retry_backoff_seconds * attempt
                            time.sleep(sleep_s)
                if content is None:
                    raise RuntimeError(f"request failed after retries: {last_error}")
        except RuntimeError as e:
            print(f"API error for context id={ctx.get('pet', {}).get('id')}: {e}")
            continue

        parsed = None
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(content[start : end + 1])
                except json.JSONDecodeError:
                    pass
        if parsed is None:
            print(f"skip: could not parse JSON from model for {ctx.get('pet', {}).get('id')}")
            continue

        errs = validate_turn(ctx, parsed)
        if errs:
            print(f"skip invalid model output for {ctx.get('pet', {}).get('id')}: {errs}")
            continue
        out_rows.append({"context": ctx, "output": parsed})

    args.cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    if not out_rows:
        raise SystemExit("No labeled rows produced.")
    write_jsonl(args.output, out_rows)
    print(f"wrote {len(out_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
