"""
Inference script for Speech Enhancement
"""

import torch
import argparse
from pathlib import Path
import sys
import yaml

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np

from models.transformer import TransformerSpeechEnhancement
from utils.audio_processing import AudioProcessor
from utils.metrics import MetricsCalculator


NOISE_GATE_THRESHOLD_DB = -35.0
NOISE_GATE_REDUCTION_DB = 18.0
NOISE_GATE_FRAME_MS = 20


def apply_noise_gate(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if audio.size == 0:
        return audio

    frame_len = int(sample_rate * (NOISE_GATE_FRAME_MS / 1000.0))
    if frame_len < 1:
        return audio

    hop = max(1, frame_len // 2)
    rms = []
    centers = []
    for start in range(0, max(1, len(audio) - frame_len + 1), hop):
        frame = audio[start:start + frame_len]
        rms.append(np.sqrt(np.mean(frame ** 2) + 1e-8))
        centers.append(start + frame_len // 2)

    rms = np.asarray(rms)
    centers = np.asarray(centers)
    rms_db = 20.0 * np.log10(rms + 1e-8)
    gain = np.where(rms_db < NOISE_GATE_THRESHOLD_DB,
                    10 ** (-NOISE_GATE_REDUCTION_DB / 20.0),
                    1.0)

    if centers[0] != 0:
        centers = np.insert(centers, 0, 0)
        gain = np.insert(gain, 0, gain[0])
    if centers[-1] < len(audio) - 1:
        centers = np.append(centers, len(audio) - 1)
        gain = np.append(gain, gain[-1])

    envelope = np.interp(np.arange(len(audio)), centers, gain)
    return audio * envelope


class SpeechEnhancer:
    """Speech enhancement inference"""
    
    def __init__(self, checkpoint_path: str, config_path: str = None):
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get config
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = checkpoint.get('config', {})
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        model_config = self.config.get('model', {})
        self.model = TransformerSpeechEnhancement(
            n_fft=model_config.get('n_fft', 1024),
            d_model=model_config.get('d_model', 512),
            nhead=model_config.get('nhead', 8),
            num_encoder_layers=model_config.get('num_encoder_layers', 6),
            dim_feedforward=model_config.get('dim_feedforward', 2048),
            dropout=model_config.get('dropout', 0.1),
            use_cooperative=model_config.get('use_cooperative', True)
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor(
            sample_rate=model_config.get('sample_rate', 16000),
            n_fft=model_config.get('n_fft', 1024),
            hop_length=model_config.get('hop_length', 256),
            win_length=model_config.get('win_length', 1024)
        )
        self.audio_processor.to(self.device)
        
        print(f"Model loaded from {checkpoint_path}")
    
    def enhance_file(
        self,
        input_path: str,
        output_path: str,
        reference_path: str = None
    ):
        """
        Enhance a single audio file
        
        Args:
            input_path: Path to noisy audio file
            output_path: Path to save enhanced audio
            reference_path: Optional path to clean reference for metrics
        """
        print(f"Processing: {input_path}")
        
        # Load audio
        noisy_audio, sr = self.audio_processor.load_audio(input_path)
        noisy_audio = noisy_audio.to(self.device)
        
        # Enhance
        with torch.no_grad():
            enhanced_audio = self.enhance(noisy_audio)
        
        # Save output
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.audio_processor.save_audio(enhanced_audio, str(output_path))
        print(f"Saved enhanced audio to: {output_path}")
        
        # Compute metrics if reference provided
        if reference_path:
            clean_audio, _ = self.audio_processor.load_audio(reference_path)
            min_len = min(clean_audio.shape[-1], enhanced_audio.shape[-1])
            
            metrics_calc = MetricsCalculator(sr)
            metrics = metrics_calc.compute_all_metrics(
                clean_audio[0, :min_len].cpu(),
                enhanced_audio[0, :min_len].cpu()
            )
            metrics_calc.print_metrics(metrics)
    
    def enhance(self, noisy_audio: torch.Tensor) -> torch.Tensor:
        """
        Enhance noisy audio
        
        Args:
            noisy_audio: Noisy audio tensor (1, samples)
        
        Returns:
            Enhanced audio tensor
        """
        # Compute STFT
        noisy_stft = self.audio_processor.compute_stft(noisy_audio)
        
        # Get magnitude and phase
        noisy_mag = torch.abs(noisy_stft)
        noisy_phase = torch.angle(noisy_stft)
        
        # Transpose for model
        noisy_mag_input = noisy_mag.transpose(1, 2)  # (B, T, F)
        
        # Predict mask
        with torch.no_grad():
            pred_mask = self.model(noisy_mag_input)
        
        # Apply mask
        enhanced_mag = noisy_mag_input * pred_mask
        enhanced_mag = enhanced_mag.transpose(1, 2)  # (B, F, T)
        
        # Reconstruct with original phase
        enhanced_stft = enhanced_mag * torch.exp(1j * noisy_phase)
        
        # Inverse STFT
        enhanced_audio = self.audio_processor.compute_istft(enhanced_stft)

        enhanced_np = enhanced_audio.squeeze(0).detach().cpu().numpy()
        enhanced_np = np.asarray(enhanced_np, dtype=np.float32)
        enhanced_np = np.nan_to_num(enhanced_np, nan=0.0, posinf=0.0, neginf=0.0)
        enhanced_np = apply_noise_gate(enhanced_np, self.audio_processor.sample_rate)

        enhanced_audio = torch.from_numpy(enhanced_np).unsqueeze(0).to(self.device)
        return enhanced_audio
    
    def enhance_directory(
        self,
        input_dir: str,
        output_dir: str,
        reference_dir: str = None
    ):
        """
        Enhance all audio files in a directory
        
        Args:
            input_dir: Directory containing noisy audio files
            output_dir: Directory to save enhanced audio files
            reference_dir: Optional directory with clean references
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all audio files
        audio_files = []
        for ext in ['.wav', '.flac', '.mp3']:
            audio_files.extend(input_dir.glob(f'*{ext}'))
        
        print(f"Found {len(audio_files)} audio files to process")
        
        for audio_file in audio_files:
            # Determine output path
            output_file = output_dir / audio_file.name
            
            # Determine reference path if provided
            reference_file = None
            if reference_dir:
                reference_file = Path(reference_dir) / audio_file.name
                if not reference_file.exists():
                    reference_file = None
            
            # Enhance
            self.enhance_file(
                str(audio_file),
                str(output_file),
                str(reference_file) if reference_file else None
            )
        
        print(f"\nProcessed all files. Enhanced audio saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Speech Enhancement Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (optional)')
    parser.add_argument('--input', type=str, required=True,
                       help='Input audio file or directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output audio file or directory')
    parser.add_argument('--reference', type=str, default=None,
                       help='Reference clean audio for metrics (optional)')
    
    args = parser.parse_args()
    
    # Initialize enhancer
    enhancer = SpeechEnhancer(args.checkpoint, args.config)
    
    # Check if input is file or directory
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file
        enhancer.enhance_file(args.input, args.output, args.reference)
    elif input_path.is_dir():
        # Directory
        enhancer.enhance_directory(args.input, args.output, args.reference)
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == '__main__':
    main()
