"""
Export Speech Enhancement Model
"""

import argparse
from pathlib import Path
import sys

import torch
import yaml

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.transformer import TransformerSpeechEnhancement


def main():
    parser = argparse.ArgumentParser(description="Export Speech Enhancement Model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional config override (defaults to checkpoint config)")
    parser.add_argument("--output", type=str, default="exports/transformer_enhancer_torchscript.pt",
                        help="Output path for TorchScript model")
    parser.add_argument("--example_frames", type=int, default=100,
                        help="Number of time frames for example input")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = checkpoint.get("config", {})

    model_config = config.get("model", {})
    n_fft = model_config.get("n_fft", 1024)

    model = TransformerSpeechEnhancement(
        n_fft=n_fft,
        d_model=model_config.get("d_model", 512),
        nhead=model_config.get("nhead", 8),
        num_encoder_layers=model_config.get("num_encoder_layers", 6),
        dim_feedforward=model_config.get("dim_feedforward", 2048),
        dropout=model_config.get("dropout", 0.1),
        use_cooperative=model_config.get("use_cooperative", True),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    example_input = torch.randn(1, args.example_frames, n_fft // 2 + 1)
    traced = torch.jit.trace(model, example_input, check_trace=False)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))

    print(f"Exported TorchScript model to: {output_path}")


if __name__ == "__main__":
    main()
