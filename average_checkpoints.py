"""Average compatible model checkpoints into one inference checkpoint."""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.inputs) < 2:
        parser.error("At least two checkpoints are required for averaging.")
    if args.output.exists():
        parser.error("Output already exists; choose a new path.")

    checkpoints = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in args.inputs
    ]
    first = checkpoints[0]
    for checkpoint in checkpoints[1:]:
        for key in ("protocol", "implementation", "config", "seed"):
            if checkpoint[key] != first[key]:
                raise ValueError(f"Checkpoint mismatch for {key}")
        if checkpoint["model"].keys() != first["model"].keys():
            raise ValueError("Checkpoint state dictionaries do not match")

    averaged = {}
    for name in first["model"]:
        tensors = [checkpoint["model"][name] for checkpoint in checkpoints]
        if tensors[0].is_floating_point():
            value = torch.stack([tensor.double() for tensor in tensors]).mean(0)
            averaged[name] = value.to(tensors[0].dtype)
        else:
            if any(not torch.equal(tensor, tensors[0]) for tensor in tensors[1:]):
                raise ValueError(f"Non-floating state differs for {name}")
            averaged[name] = tensors[0]

    def checkpoint_step(checkpoint):
        if "selected_step" in checkpoint:
            return int(checkpoint["selected_step"])
        training_config = checkpoint.get("training_config") or {}
        batch_size = int(training_config.get("batch_size", 32))
        return int(checkpoint["train_tokens"] // (batch_size * 256))

    steps = [checkpoint_step(checkpoint) for checkpoint in checkpoints]
    output = {
        "protocol": first["protocol"],
        "implementation": first["implementation"],
        "config": first["config"],
        "model": averaged,
        "seed": first["seed"],
        "train_tokens": max(checkpoint["train_tokens"] for checkpoint in checkpoints),
        "selected_step": max(steps),
        "training_config": first.get("training_config"),
        "averaged_steps": steps,
        "averaged_inputs": [str(path) for path in args.inputs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    summary = {
        "output": str(args.output),
        "inputs": [str(path) for path in args.inputs],
        "averaged_steps": steps,
        "train_tokens": output["train_tokens"],
    }
    args.output.with_suffix(".txt").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
