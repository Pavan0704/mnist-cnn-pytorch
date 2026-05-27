"""
main.py — One-shot runner: train → evaluate → summarise
=========================================================
Run this single file to reproduce the full experiment.

    python main.py                  # default 10 epochs
    python main.py --epochs 5       # quick smoke-test
"""

import argparse
from train    import main as train_main
from evaluate import main as eval_main


def parse_args():
    p = argparse.ArgumentParser(description="MNIST CNN — full pipeline")
    p.add_argument("--epochs",         type=int,   default=10)
    p.add_argument("--batch-size",     type=int,   default=64)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--data-dir",       type=str,   default="data")
    p.add_argument("--results-dir",    type=str,   default="results")
    p.add_argument("--checkpoint-dir", type=str,   default="checkpoints")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("\n>>  Starting full MNIST CNN experiment ...\n")
    train_main(args)
    eval_main(args)
    print("[DONE]  All done!  Check the results/ folder for plots & reports.\n")
