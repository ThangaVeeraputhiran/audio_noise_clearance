"""
Dataset class for Speech Enhancement
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os
import random
from pathlib import Path
from typing import Tuple, List, Optional
import torchaudio


class SpeechEnhancementDataset(Dataset):
    """
    Dataset for speech enhancement training
    Loads clean and noisy audio pairs
    """
    
    def __init__(
        self,
        clean_dir: str,
        noisy_dir: Optional[str] = None,
        noise_dir: Optional[str] = None,
        sample_rate: int = 16000,
        segment_length: float = 4.0,
        snr_range: Tuple[float, float] = (-5, 20),
        transform=None,
        mode: str = 'train'
    ):
        """
        Args:
            clean_dir: Directory containing clean audio files
            noisy_dir: Directory containing pre-mixed noisy audio (optional)
            noise_dir: Directory containing noise files for on-the-fly mixing (optional)
            sample_rate: Target sample rate
            segment_length: Length of audio segments in seconds
            snr_range: SNR range for noise mixing (min_snr, max_snr) in dB
            transform: Optional transform to apply
            mode: 'train', 'val', or 'test'
        """
        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir) if noisy_dir else None
        self.noise_dir = Path(noise_dir) if noise_dir else None
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.segment_samples = int(segment_length * sample_rate)
        self.snr_range = snr_range
        self.transform = transform
        self.mode = mode
        
        # Get file lists
        self.clean_files = self._get_audio_files(self.clean_dir)
        
        if self.noisy_dir and self.noisy_dir.exists():
            self.noisy_files = self._get_audio_files(self.noisy_dir)
            self.use_premixed = True
        else:
            self.noisy_files = None
            self.use_premixed = False
        
        if self.noise_dir and self.noise_dir.exists():
            self.noise_files = self._get_audio_files(self.noise_dir)
        else:
            self.noise_files = []
        
        print(f"Dataset loaded: {len(self.clean_files)} clean files")
        if self.use_premixed:
            print(f"Using {len(self.noisy_files)} pre-mixed noisy files")
        elif self.noise_files:
            print(f"Using {len(self.noise_files)} noise files for on-the-fly mixing")
    
    def _get_audio_files(self, directory: Path) -> List[Path]:
        """Get all audio files from directory"""
        audio_extensions = {'.wav', '.flac', '.mp3', '.ogg'}
        files = []
        for ext in audio_extensions:
            files.extend(directory.rglob(f'*{ext}'))
        return sorted(files)
    
    def _load_audio(self, file_path: Path) -> torch.Tensor:
        """Load and preprocess audio file"""
        import soundfile as sf
        
        # Load with soundfile directly
        waveform, sr = sf.read(str(file_path))
        waveform = torch.from_numpy(waveform).float()
        
        # Ensure 2D shape (channels, samples)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.T
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform
    
    def _random_segment(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract random segment from audio"""
        if waveform.shape[-1] > self.segment_samples:
            max_start = waveform.shape[-1] - self.segment_samples
            start = random.randint(0, max_start)
            segment = waveform[:, start:start + self.segment_samples]
        else:
            # Pad if too short
            segment = torch.nn.functional.pad(
                waveform,
                (0, self.segment_samples - waveform.shape[-1])
            )
        return segment
    
    def _add_noise(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        snr_db: float
    ) -> torch.Tensor:
        """Add noise to clean audio at specified SNR"""
        # Match lengths
        if noise.shape[-1] < clean.shape[-1]:
            # Repeat noise if it's shorter
            repeats = (clean.shape[-1] // noise.shape[-1]) + 1
            noise = noise.repeat(1, repeats)
        
        # Random segment from noise
        if noise.shape[-1] > clean.shape[-1]:
            start = random.randint(0, noise.shape[-1] - clean.shape[-1])
            noise = noise[:, start:start + clean.shape[-1]]
        
        # Calculate powers
        clean_power = torch.mean(clean ** 2)
        noise_power = torch.mean(noise ** 2)
        
        # Calculate scaling factor
        snr_linear = 10 ** (snr_db / 10)
        scale = torch.sqrt(clean_power / (noise_power * snr_linear + 1e-8))
        
        # Mix
        noisy = clean + scale * noise
        
        return noisy
    
    def __len__(self) -> int:
        return len(self.clean_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sample
        
        Returns:
            clean: Clean audio segment
            noisy: Noisy audio segment
        """
        # Load clean audio
        clean = self._load_audio(self.clean_files[idx])
        clean = self._random_segment(clean)
        
        # Get or create noisy audio
        if self.use_premixed:
            # Use pre-mixed noisy file
            noisy_idx = idx % len(self.noisy_files)
            noisy = self._load_audio(self.noisy_files[noisy_idx])
            noisy = self._random_segment(noisy)
        else:
            # Create noisy audio on-the-fly
            if self.noise_files:
                noise_idx = random.randint(0, len(self.noise_files) - 1)
                noise = self._load_audio(self.noise_files[noise_idx])
                snr_db = random.uniform(*self.snr_range)
                noisy = self._add_noise(clean, noise, snr_db)
            else:
                # No noise available, use clean as noisy (for testing)
                noisy = clean.clone()
        
        # Apply transforms if any
        if self.transform:
            clean = self.transform(clean)
            noisy = self.transform(noisy)
        
        # Remove channel dimension
        clean = clean.squeeze(0)
        noisy = noisy.squeeze(0)
        
        return noisy, clean


def create_dataloaders(
    clean_dir: str,
    noisy_dir: Optional[str] = None,
    noise_dir: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create full dataset
    full_dataset = SpeechEnhancementDataset(
        clean_dir=clean_dir,
        noisy_dir=noisy_dir,
        noise_dir=noise_dir,
        **kwargs
    )
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
