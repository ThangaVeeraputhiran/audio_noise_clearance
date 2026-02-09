"""
Audio Processing Utilities for Speech Enhancement
"""

import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional


class AudioProcessor:
    """Handles audio loading, preprocessing, and feature extraction"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 128
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        
        # Initialize transforms
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=None
        )
        
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            n_stft=n_fft // 2 + 1
        )
        
        self.inverse_stft = torchaudio.transforms.InverseSpectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length
        )
    
    def to(self, device):
        """Move all transforms to the specified device"""
        self.stft = self.stft.to(device)
        self.mel_scale = self.mel_scale.to(device)
        self.inverse_stft = self.inverse_stft.to(device)
        return self
    
    def load_audio(self, file_path: str, target_sr: Optional[int] = None) -> Tuple[torch.Tensor, int]:
        """Load audio file and resample if needed"""
        waveform, sr = torchaudio.load(file_path)
        
        target_sr = target_sr or self.sample_rate
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
            sr = target_sr
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform, sr
    
    def save_audio(self, waveform: torch.Tensor, file_path: str, sample_rate: int = None):
        """Save audio to file"""
        sample_rate = sample_rate or self.sample_rate
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        sf.write(file_path, waveform.T, sample_rate)
    
    def compute_stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute Short-Time Fourier Transform"""
        return self.stft(waveform)
    
    def compute_istft(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Compute Inverse Short-Time Fourier Transform"""
        return self.inverse_stft(spectrogram)
    
    def compute_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute mel-spectrogram"""
        stft = self.compute_stft(waveform)
        magnitude = torch.abs(stft)
        mel_spec = self.mel_scale(magnitude)
        return mel_spec
    
    def extract_features(self, waveform: torch.Tensor) -> dict:
        """Extract multiple features from audio"""
        stft = self.compute_stft(waveform)
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Log magnitude
        log_magnitude = torch.log(magnitude + 1e-8)
        
        # Mel spectrogram
        mel_spec = self.mel_scale(magnitude)
        log_mel = torch.log(mel_spec + 1e-8)
        
        return {
            'stft': stft,
            'magnitude': magnitude,
            'phase': phase,
            'log_magnitude': log_magnitude,
            'mel_spectrogram': mel_spec,
            'log_mel': log_mel
        }
    
    def normalize_audio(self, waveform: torch.Tensor, method: str = 'peak') -> torch.Tensor:
        """Normalize audio waveform"""
        if method == 'peak':
            max_val = torch.max(torch.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val
        elif method == 'rms':
            rms = torch.sqrt(torch.mean(waveform ** 2))
            if rms > 0:
                waveform = waveform / (rms * 10)
        return waveform
    
    def add_noise(
        self,
        clean_audio: torch.Tensor,
        noise_audio: torch.Tensor,
        snr_db: float
    ) -> torch.Tensor:
        """Add noise to clean audio at specified SNR"""
        # Ensure same length
        if clean_audio.shape[-1] != noise_audio.shape[-1]:
            min_len = min(clean_audio.shape[-1], noise_audio.shape[-1])
            clean_audio = clean_audio[..., :min_len]
            noise_audio = noise_audio[..., :min_len]
        
        # Calculate current power
        clean_power = torch.mean(clean_audio ** 2)
        noise_power = torch.mean(noise_audio ** 2)
        
        # Calculate scaling factor for desired SNR
        snr_linear = 10 ** (snr_db / 10)
        scale = torch.sqrt(clean_power / (noise_power * snr_linear))
        
        # Add scaled noise
        noisy_audio = clean_audio + scale * noise_audio
        
        return noisy_audio
    
    def segment_audio(
        self,
        waveform: torch.Tensor,
        segment_length: float,
        overlap: float = 0.5
    ) -> list:
        """Segment audio into overlapping chunks"""
        sr = self.sample_rate
        segment_samples = int(segment_length * sr)
        hop_samples = int(segment_samples * (1 - overlap))
        
        segments = []
        start = 0
        while start + segment_samples <= waveform.shape[-1]:
            segment = waveform[..., start:start + segment_samples]
            segments.append(segment)
            start += hop_samples
        
        # Add last segment if there's remaining audio
        if start < waveform.shape[-1]:
            last_segment = waveform[..., -segment_samples:]
            segments.append(last_segment)
        
        return segments


def compute_snr(clean: torch.Tensor, noisy: torch.Tensor) -> float:
    """Compute Signal-to-Noise Ratio"""
    noise = noisy - clean
    signal_power = torch.mean(clean ** 2)
    noise_power = torch.mean(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()


def compute_si_snr(clean: torch.Tensor, enhanced: torch.Tensor) -> float:
    """Compute Scale-Invariant Signal-to-Noise Ratio"""
    # Zero-mean
    clean = clean - torch.mean(clean)
    enhanced = enhanced - torch.mean(enhanced)
    
    # Compute projection
    s_target = torch.sum(clean * enhanced) * clean / (torch.sum(clean ** 2) + 1e-8)
    e_noise = enhanced - s_target
    
    # Compute SI-SNR
    si_snr = 10 * torch.log10(
        torch.sum(s_target ** 2) / (torch.sum(e_noise ** 2) + 1e-8)
    )
    
    return si_snr.item()
