from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from aipet_distill.constants import NEED_KEYS
from aipet_distill.schemas import GameContext, PetTurn


def _ids_by_type(context: dict[str, Any]) -> dict[str, set[str]]:
    scene = context.get("scene") or {}
    out: dict[str, set[str]] = {"player": set(), "pet": set(), "object": set()}
    for p in scene.get("players") or []:
        if isinstance(p, dict) and "id" in p:
            out["player"].add(p["id"])
    for o in scene.get("objects") or []:
        if isinstance(o, dict) and "id" in o:
            out["object"].add(o["id"])
    for op in scene.get("other_pets") or []:
        if isinstance(op, dict) and "id" in op:
            out["pet"].add(op["id"])
    return out


def validate_turn(context: dict[str, Any], output: dict[str, Any]) -> list[str]:
    """Return a list of human-readable errors; empty means valid."""
    errors: list[str] = []
    try:
        GameContext.model_validate(context)
    except ValidationError as e:
        return [f"context: {e}"]

    try:
        turn = PetTurn.model_validate(output)
    except ValidationError as e:
        return [f"output: {e}"]

    if turn.decision.primary_need not in NEED_KEYS:
        errors.append(f"primary_need must be one of {NEED_KEYS}, got {turn.decision.primary_need!r}")

    if turn.decision.action.need != turn.decision.primary_need:
        errors.append("decision.action.need must equal decision.primary_need")

    allowed = (context.get("constraints") or {}).get("allowed_actions_by_need") or {}
    need = turn.decision.action.need
    legal = allowed.get(need) or []
    if turn.decision.action.name not in legal:
        errors.append(
            f"action.name {turn.decision.action.name!r} not in allowed_actions_by_need[{need!r}]: {legal!r}"
        )

    ids = _ids_by_type(context)
    tgt = turn.decision.action.target
    if tgt is not None:
        if tgt.type not in ids:
            errors.append(f"target.type invalid: {tgt.type!r}")
        elif tgt.id not in ids[tgt.type]:
            errors.append(f"target.id {tgt.id!r} not found for type {tgt.type!r}")

    return errors
