from __future__ import annotations

from .evaluate import run_evaluate_stage
from .train import run_train_stage
from .tune import run_tune_stage


def run_all() -> dict:
    train_out = run_train_stage()
    tune_out = run_tune_stage()
    eval_out = run_evaluate_stage()
    return {
        "train": train_out,
        "tune": tune_out,
        "evaluate": eval_out,
    }


if __name__ == "__main__":
    result = run_all()
    print(result)
