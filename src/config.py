"""
Central config — tweak these to change model size, paths, and training behaviour.
"""
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


@dataclass
class ModelConfig:
    base_model: str = "HuggingFaceTB/SmolLM2-135M"  # swap for larger if needed
    max_input_length: int = 128
    max_output_length: int = 64


@dataclass
class LoraConfig:
    r: int = 8
    lora_alpha: int = 16
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"


@dataclass
class TrainConfig:
    output_dir: str = str(BASE_DIR / "models" / "lora_adapter")
    num_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-4
    warmup_steps: int = 50
    save_steps: int = 100
    logging_steps: int = 10
    fp16: bool = False  # set True if GPU available


model_cfg = ModelConfig()
lora_cfg = LoraConfig()
train_cfg = TrainConfig()
