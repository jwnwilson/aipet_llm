"""Eval entry point for VastAI pods — delegates to the RunPod eval script.

VastAI and RunPod use the same S3-backed eval flow; the logic lives in one place.
"""
from adapters.compute.runpod.eval_script import main

if __name__ == "__main__":
    main()
