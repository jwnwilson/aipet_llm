"""Compact text format: simple lines in, need|action|say out."""

from __future__ import annotations

from typing import Any

from aipet_distill.constants import NEED_KEYS

SEP = "|"
LABEL_SEP = "||"


def game_context_to_simple_text(context: dict[str, Any]) -> str:
    """Encode GameContext dict into fixed-layout lines (fewer tokens than full JSON)."""
    pet = context.get("pet") or {}
    needs = context.get("needs") or {}
    scene = context.get("scene") or {}
    constraints = context.get("constraints") or {}
    allowed = (constraints.get("allowed_actions_by_need") or {}).copy()

    pet_line = "PET: {id} {name} {mood}".format(
        id=pet.get("id", ""),
        name=pet.get("name", ""),
        mood=(pet.get("mood_hint") or "").replace("\n", " ").strip(),
    )

    def _n(key: str) -> int:
        v = needs.get(key)
        return int(v) if v is not None else 0

    needs_line = "NEEDS: h={h} t={t} b={b} s={s} toi={toi}".format(
        h=_n("hunger"),
        t=_n("tiredness"),
        b=_n("boredom"),
        s=_n("social"),
        toi=_n("toilet"),
    )

    scene_parts: list[str] = []
    for ob in scene.get("objects") or []:
        if not isinstance(ob, dict):
            continue
        label = (ob.get("name") or ob.get("id") or "obj").replace(" ", "_")
        dm = ob.get("distance_m", 0)
        scene_parts.append(f"{label}@{dm}m")
    for pl in scene.get("players") or []:
        if not isinstance(pl, dict):
            continue
        nm = pl.get("display_name") or pl.get("id") or "player"
        rel = pl.get("relationship") or "?"
        dm = pl.get("distance_m", 0)
        busy = "busy" if pl.get("busy") else "free"
        scene_parts.append(f"{nm}({rel},{dm}m,{busy})")
    for op in scene.get("other_pets") or []:
        if not isinstance(op, dict):
            continue
        nm = op.get("name") or op.get("id") or "pet"
        dm = op.get("distance_m", 0)
        friendly = "friend" if op.get("friendly") else "neutral"
        scene_parts.append(f"{nm}@pet@{dm}m,{friendly}")

    scene_line = "SCENE: " + " ".join(scene_parts)

    action_chunks: list[str] = []
    for nk in NEED_KEYS:
        acts = allowed.get(nk) or []
        inner = ",".join(acts)
        action_chunks.append(f"{nk}:[{inner}]")

    actions_line = "Actions: " + "; ".join(action_chunks)

    task_line = "TASK: pick one need and one action."

    return "\n".join([pet_line, needs_line, scene_line, actions_line, task_line])


def build_say_prompt(simple_text: str, need: str, action: str) -> str:
    """Prompt for stage-2 LM: only generates the spoken line."""
    return (
        f"{simple_text}\n"
        f"Chosen need: {need}\n"
        f"Chosen action: {action}\n"
        "SAY:"
    )


def parse_need_action_say(raw: str) -> tuple[str, str, str] | None:
    """Parse 'need|action|say' (say may contain | if we take after 2nd pipe only)."""
    text = raw.strip()
    for prefix in ("SAY:", "say:", "Output:", "output:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    # strip code fences / quotes
    text = text.strip().strip("`").strip()

    parts = text.split(SEP, 2)
    if len(parts) < 3:
        return None
    need, action, say = parts[0].strip(), parts[1].strip(), parts[2].strip()
    if not need or not action:
        return None
    say = say.split("\n")[0].strip()
    return need, action, say


def _legal_actions_for(context: dict[str, Any], need: str) -> list[str]:
    constraints = context.get("constraints") or {}
    allowed = constraints.get("allowed_actions_by_need") or {}
    return list(allowed.get(need) or [])


def validate_simple_turn(context: dict[str, Any], need: str, action: str) -> list[str]:
    errs: list[str] = []
    if need not in NEED_KEYS:
        errs.append(f"need must be one of {NEED_KEYS}, got {need!r}")
        return errs
    legal = _legal_actions_for(context, need)
    if not legal:
        errs.append(f"no allowed actions for need {need!r}")
    elif action not in legal:
        errs.append(f"action {action!r} not in allowed_actions_by_need[{need!r}]: {legal!r}")
    return errs


def clamp_need_action(context: dict[str, Any], need: str, action: str) -> tuple[str, str]:
    """If prediction is illegal, fall back to first legal action for that need, else max-need heuristic."""
    if need in NEED_KEYS:
        legal = _legal_actions_for(context, need)
        if action in legal:
            return need, action
        if legal:
            return need, legal[0]

    needs = context.get("needs") or {}
    scores: list[tuple[int, str]] = []
    for nk in NEED_KEYS:
        v = needs.get(nk)
        if v is not None:
            scores.append((int(v), nk))
    if not scores:
        nk = "boredom"
    else:
        nk = max(scores, key=lambda x: x[0])[1]

    legal2 = _legal_actions_for(context, nk)
    act = legal2[0] if legal2 else "explore"
    return nk, act


def format_line(need: str, action: str, say: str) -> str:
    """Single line response for APIs and training targets."""
    # escape pipes in say? User asked simple format — forbid newlines in say
    say_one = say.replace("\n", " ").replace(SEP, " ").strip()
    return f"{need}{SEP}{action}{SEP}{say_one}"
