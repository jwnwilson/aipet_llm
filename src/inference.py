"""
Fast inference with response cache.
Usage:
    from src.inference import GameAI
    ai = GameAI()
    print(ai.respond("[NPC:ATTACK] Player approaches."))
"""
import time
from functools import lru_cache
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.config import model_cfg, train_cfg

ADAPTER_PATH = Path(train_cfg.output_dir)


class GameAI:
    def __init__(self, use_adapter: bool = True):
        print("Loading model…")
        t0 = time.time()

        tok_path = (
            ADAPTER_PATH
            if use_adapter and ADAPTER_PATH.exists()
            else model_cfg.base_model
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            model_cfg.base_model,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

        if use_adapter and ADAPTER_PATH.exists():
            self.model = PeftModel.from_pretrained(base, str(ADAPTER_PATH))
            print("LoRA adapter loaded.")
        else:
            self.model = base
            print("Base model loaded (no adapter found).")

        self.model.eval()

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=model_cfg.max_output_length,
            do_sample=False,  # greedy = deterministic + fastest
            pad_token_id=self.tokenizer.eos_token_id,
        )
        print(f"Ready in {time.time() - t0:.1f}s")

    @lru_cache(maxsize=512)  # repeated NPC triggers → ~0 ms
    def respond(self, prompt: str) -> str:
        t0 = time.time()
        out = self.pipe(prompt)[0]["generated_text"]
        # strip the echoed prompt and take the first line
        response = out[len(prompt):].strip().split("\n")[0]
        ms = (time.time() - t0) * 1000
        print(f"[{ms:.0f}ms] {prompt!r} → {response!r}")
        return response


if __name__ == "__main__":
    ai = GameAI(use_adapter=False)  # use base model for quick demo

    test_prompts = [
        "[NPC:ATTACK] Player approaches.",
        "[NPC:TRADE] Player approaches.",
        "[NPC:QUEST_HINT] Player approaches.",
        "[NPC:TAUNT] Player approaches.",
        "[NPC:ATTACK] Player approaches.",  # second call — hits cache
    ]
    for p in test_prompts:
        ai.respond(p)
