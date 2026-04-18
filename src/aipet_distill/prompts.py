from __future__ import annotations

import json
from typing import Any

SYSTEM = """You are the brain of a virtual pet in a multiplayer game.
You receive ONE JSON object (the game context) describing the pet, needs, scene, and legal actions.
You MUST reply with ONE JSON object only — no markdown, no code fences, no commentary.
The JSON must match this shape:
{
  "decision": {
    "primary_need": "<one of hunger|tiredness|boredom|social|toilet>",
    "action": {
      "need": "<same as primary_need>",
      "name": "<must be one of constraints.allowed_actions_by_need[need]>",
      "target": null OR {"type":"player|pet|object","id":"<existing id from scene>"}
    }
  },
  "needs_after_intent": { "hunger": 0-100, "tiredness": 0-100, "boredom": 0-100, "social": 0-100, "toilet": 0-100 },
  "emotion": { "label": "<snake_case label>", "intensity": 0-1, "reason": "<short>" },
  "dialog": "<short in-character line>",
  "confidence": 0-1
}
Use "target" when the action clearly applies to a specific player, pet, or object in the scene."""


def format_context_json(context: dict[str, Any] | Any) -> str:
    if hasattr(context, "model_dump"):
        data = context.model_dump()
    else:
        data = context
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def build_training_text(context: dict[str, Any] | Any, response_json: str) -> str:
    ctx = format_context_json(context)
    return (
        f"<|system|>\n{SYSTEM}\n"
        f"<|context|>\n{ctx}\n"
        f"<|response|>\n{response_json.strip()}\n"
    )


def build_prompt_text(context: dict[str, Any] | Any) -> str:
    ctx = format_context_json(context)
    return f"<|system|>\n{SYSTEM}\n<|context|>\n{ctx}\n<|response|>\n"
