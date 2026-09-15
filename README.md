# train_utils

A modular, reusable fine-tuning framework for LLMs and VLMs on a local RTX 4090.
Supports SFT and DPO (extensible to GRPO, PPO, etc.) via a single YAML config.

Built on PyTorch, [Unsloth](https://github.com/unslothai/unsloth), HuggingFace Transformers, TRL, and W&B.

---

## Installation

Install the package in editable mode once from the `all_in` root.
After this, `train_utils` is importable from any project on the machine.

```bash
cd ~/Projects/all_in
pip install -e .
```

Install Unsloth separately with the correct CUDA + PyTorch variant for your system:

```bash
# CUDA 12.1 + PyTorch 2.3.0 (recommended for RTX 4090)
pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"
```

---

## Running a training job

### SFT (Supervised Fine-Tuning)

```bash
train configs/sft_example.yaml
```

### DPO (Direct Preference Optimization)

```bash
train configs/dpo_example.yaml
```

### As a Python module

```bash
python -m train_utils.train configs/sft_example.yaml
```

### From another project

```python
from train_utils.train import run

run("/path/to/my_project/configs/run.yaml")
```

---

## Project layout

```
all_in/
├── pyproject.toml
├── requirements.txt
├── configs/                        # Example YAML run configs
│   ├── sft_example.yaml
│   └── dpo_example.yaml
└── train_utils/
    ├── train.py                    # CLI entry point
    ├── configs/                    # Typed dataclass configs
    │   ├── base_config.py          # RunConfig + TrainingArgs
    │   ├── model_config.py         # ModelConfig, LoRAConfig, QuantizationConfig
    │   ├── sft_config.py           # SFTConfig
    │   └── dpo_config.py           # DPOConfig
    ├── models/
    │   ├── loader.py               # load_model_and_tokenizer()
    │   └── lora.py                 # apply_lora(), save_lora_adapter()
    ├── data/
    │   ├── base_dataset.py         # Abstract BaseDataset
    │   ├── sft_dataset.py          # SFTDataset
    │   └── dpo_dataset.py          # DPODataset
    ├── trainers/
    │   ├── base_trainer.py         # Abstract BaseTrainer
    │   ├── sft_trainer.py          # SFTTrainer (wraps TRL SFTTrainer)
    │   ├── dpo_trainer.py          # DPOTrainer (wraps TRL DPOTrainer)
    │   └── registry.py             # TRAINER_REGISTRY + build_trainer()
    └── utils/
        ├── gpu_utils.py            # VRAM logging, bf16 detection, device setup
        └── yaml_utils.py           # load_yaml_config() → RunConfig
```

---

## Using train_utils from another project

Install once (see above), then create a YAML config and call `run()`:

```
~/Projects/my_robot_vla/
├── configs/
│   └── run.yaml          # points to your HF model, dataset, output dir, etc.
├── data/
│   └── my_dataset.py     # optional: subclass BaseDataset for custom data
└── train.py
```

`train.py`:

```python
from train_utils.train import run

run("configs/run.yaml")
```

### Custom dataset

Subclass `BaseDataset` to plug in any data source without touching trainer code:

```python
from datasets import Dataset
from train_utils.data.base_dataset import BaseDataset
from train_utils.configs.base_config import RunConfig
from transformers import PreTrainedTokenizerBase


class MyDataset(BaseDataset):
    def __init__(self, config: RunConfig, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__(config, tokenizer)

    def get_train_dataset(self) -> Dataset:
        # load and return your dataset here
        ...

    def get_eval_dataset(self) -> Dataset | None:
        return None
```

Then pass it directly to `build_trainer`:

```python
from train_utils.models.loader import load_model_and_tokenizer
from train_utils.models.lora import apply_lora
from train_utils.trainers.registry import build_trainer
from train_utils.utils.yaml_utils import load_yaml_config

config = load_yaml_config("configs/run.yaml")
model, tokenizer = load_model_and_tokenizer(config.model)
model = apply_lora(model, config.model.lora, use_unsloth=config.model.use_unsloth)

dataset = MyDataset(config, tokenizer)
trainer = build_trainer(config, model, tokenizer, dataset)
trainer.train()
```

---

## Adding a new training algorithm

1. Create `train_utils/trainers/grpo_trainer.py` subclassing `BaseTrainer`.
2. Implement `_build_hf_trainer()` to return the configured TRL/HF trainer.
3. Add one line to `train_utils/trainers/registry.py`:

```python
from train_utils.trainers.grpo_trainer import GRPOTrainer

TRAINER_REGISTRY: dict[str, type[BaseTrainer]] = {
    "sft": SFTTrainer,
    "dpo": DPOTrainer,
    "grpo": GRPOTrainer,   # <-- new entry
}
```

4. Set `trainer_type: grpo` in your YAML config. No other changes required.

---

## W&B setup

Log in once:

```bash
wandb login
```

Set your project and entity in the YAML config:

```yaml
wandb_project: my_project
wandb_entity: my_wandb_username   # or null to use the default
training:
  report_to: wandb
```

Set `report_to: none` to disable W&B for a run.

---

## RTX 4090 recommended settings

| Setting | Recommended value | Reason |
|---|---|---|
| `model.use_unsloth` | `true` | 2× faster training, ~50% less VRAM |
| `model.attn_implementation` | `flash_attention_2` | Faster attention via Unsloth kernels |
| `model.quantization.load_in_4bit` | `true` | Fits large models in 24GB VRAM |
| `training.bf16` | `true` | Native bfloat16 on Ampere (RTX 40xx) |
| `training.gradient_checkpointing` | `true` | Trades compute for VRAM |
| `training.gradient_accumulation_steps` | `4`–`8` | Effective batch size without OOM |
