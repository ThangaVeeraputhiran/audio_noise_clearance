"""
Evaluation script for Speech Enhancement Model
"""

import argparse
from pathlib import Path
import sys

import torch
import yaml
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.transformer import TransformerSpeechEnhancement
from utils.audio_processing import AudioProcessor
from utils.dataset import create_dataloaders
from utils.metrics import AverageMeter, MetricsCalculator, _PESQ_AVAILABLE, _STOI_AVAILABLE


def build_model(config, device):
    model_config = config["model"]
    model = TransformerSpeechEnhancement(
        n_fft=model_config["n_fft"],
        d_model=model_config["d_model"],
        nhead=model_config["nhead"],
        num_encoder_layers=model_config["num_encoder_layers"],
        dim_feedforward=model_config["dim_feedforward"],
        dropout=model_config["dropout"],
        use_cooperative=model_config["use_cooperative"],
    ).to(device)
    return model


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate Speech Enhancement Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest_checkpoint.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Optional limit on number of test batches")
    parser.add_argument("--save_limit", type=int, default=10,
                        help="Max number of enhanced samples to save")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        config["hardware"]["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["seed"])

    model = build_model(config, device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    audio_processor = AudioProcessor(
        sample_rate=config["data"]["sample_rate"],
        n_fft=config["model"]["n_fft"],
        hop_length=config["model"]["hop_length"],
        win_length=config["model"]["win_length"],
    )
    audio_processor.to(device)

    _, _, test_loader = create_dataloaders(
        clean_dir=config["data"]["clean_dir"],
        noise_dir=config["data"]["noise_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["hardware"]["num_workers"],
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        sample_rate=config["data"]["sample_rate"],
        segment_length=config["data"]["segment_length"],
        snr_range=tuple(config["data"]["augmentation"]["noise_snr_range"]),
    )

    metrics_list = config.get("evaluation", {}).get("metrics", [])
    filtered_metrics = []
    for metric in metrics_list:
        if metric == "pesq" and not _PESQ_AVAILABLE:
            continue
        if metric in ("stoi", "estoi") and not _STOI_AVAILABLE:
            continue
        filtered_metrics.append(metric)

    if not filtered_metrics:
        print("No metrics available to compute. Install 'pesq' and/or 'pystoi' to enable them.")

    metrics_calc = MetricsCalculator(sample_rate=config["data"]["sample_rate"])
    meters = {name: AverageMeter() for name in filtered_metrics}

    save_enhanced = config.get("evaluation", {}).get("save_enhanced", False)
    enhanced_dir = Path(config.get("evaluation", {}).get("enhanced_dir", "data/enhanced"))
    if save_enhanced:
        enhanced_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    sample_index = 0

    with torch.no_grad():
        for batch_idx, (noisy, clean) in enumerate(tqdm(test_loader, desc="Evaluating")):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            noisy = noisy.to(device)
            clean = clean.to(device)

            noisy_stft = audio_processor.compute_stft(noisy)
            noisy_mag = torch.abs(noisy_stft).transpose(1, 2)

            pred_mask = model(noisy_mag)
            enhanced_mag = noisy_mag * pred_mask

            enhanced_stft = enhanced_mag.transpose(1, 2) * torch.exp(1j * torch.angle(noisy_stft))
            enhanced_audio = audio_processor.compute_istft(enhanced_stft)

            batch_size = clean.size(0)
            for i in range(batch_size):
                clean_i = clean[i].cpu()
                enhanced_i = enhanced_audio[i].cpu()

                if filtered_metrics:
                    if "pesq" in filtered_metrics:
                        meters["pesq"].update(metrics_calc.compute_pesq(clean_i, enhanced_i))
                    if "stoi" in filtered_metrics:
                        meters["stoi"].update(metrics_calc.compute_stoi(clean_i, enhanced_i))
                    if "estoi" in filtered_metrics:
                        meters["estoi"].update(metrics_calc.compute_stoi(clean_i, enhanced_i, extended=True))
                    if "si_snr" in filtered_metrics:
                        meters["si_snr"].update(metrics_calc.compute_si_snr(clean_i, enhanced_i))
                    if "sdr" in filtered_metrics:
                        meters["sdr"].update(metrics_calc.compute_sdr(clean_i, enhanced_i))

                if save_enhanced and saved_count < args.save_limit:
                    out_path = enhanced_dir / f"enhanced_{sample_index:05d}.wav"
                    audio_processor.save_audio(enhanced_i, str(out_path))
                    saved_count += 1

                sample_index += 1

    if filtered_metrics:
        print("\nEvaluation Results")
        print("=" * 50)
        for name, meter in meters.items():
            print(f"{name.upper():10s}: {meter.avg:7.4f}")
        print("=" * 50)

    if save_enhanced:
        print(f"Saved {saved_count} enhanced samples to: {enhanced_dir}")


if __name__ == "__main__":
    main()
