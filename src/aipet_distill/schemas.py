from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from aipet_distill.constants import NEED_KEYS


class PetInfo(BaseModel):
    id: str
    name: str
    species: str
    mood_hint: str


class NeedsState(BaseModel):
    hunger: int = Field(ge=0, le=100)
    tiredness: int = Field(ge=0, le=100)
    boredom: int = Field(ge=0, le=100)
    social: int = Field(ge=0, le=100)
    toilet: int = Field(ge=0, le=100)


class SceneObject(BaseModel):
    id: str
    name: str
    kind: str
    interactable: bool
    distance_m: float


class ScenePlayer(BaseModel):
    id: str
    display_name: str
    distance_m: float
    relationship: str
    busy: bool


class OtherPet(BaseModel):
    id: str
    name: str
    species: str
    distance_m: float
    friendly: bool


class Scene(BaseModel):
    id: str
    tags: list[str]
    objects: list[SceneObject]
    players: list[ScenePlayer]
    other_pets: list[OtherPet]


class Constraints(BaseModel):
    allowed_actions_by_need: dict[str, list[str]]


class GameContext(BaseModel):
    pet: PetInfo
    needs: NeedsState
    scene: Scene
    constraints: Constraints
    task: str

    @field_validator("constraints")
    @classmethod
    def needs_have_allowlists(cls, v: Constraints) -> Constraints:
        missing = [k for k in NEED_KEYS if k not in v.allowed_actions_by_need]
        if missing:
            raise ValueError(f"allowed_actions_by_need missing keys: {missing}")
        return v


TargetType = Literal["player", "pet", "object"]


class ActionTarget(BaseModel):
    type: TargetType
    id: str


class ActionDecision(BaseModel):
    need: str
    name: str
    target: ActionTarget | None = None


class Decision(BaseModel):
    primary_need: str
    action: ActionDecision


class EmotionOut(BaseModel):
    label: str
    intensity: float = Field(ge=0.0, le=1.0)
    reason: str


class PetTurn(BaseModel):
    decision: Decision
    needs_after_intent: dict[str, int]
    emotion: EmotionOut
    dialog: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("needs_after_intent")
    @classmethod
    def all_need_keys(cls, v: dict[str, int]) -> dict[str, int]:
        missing = [k for k in NEED_KEYS if k not in v]
        if missing:
            raise ValueError(f"needs_after_intent missing keys: {missing}")
        extra = [k for k in v if k not in NEED_KEYS]
        if extra:
            raise ValueError(f"needs_after_intent unknown keys: {extra}")
        for k, val in v.items():
            if not 0 <= val <= 100:
                raise ValueError(f"needs_after_intent[{k}] out of range: {val}")
        return v
