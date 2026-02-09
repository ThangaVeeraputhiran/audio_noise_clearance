"""
Evaluation Metrics for Speech Enhancement
"""

import torch
import numpy as np
from typing import Union, Tuple

try:
    from pesq import pesq
    _PESQ_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    pesq = None
    _PESQ_AVAILABLE = False

try:
    from pystoi import stoi
    _STOI_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    stoi = None
    _STOI_AVAILABLE = False


class MetricsCalculator:
    """Calculate various audio quality metrics"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def _to_numpy(self, audio: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert tensor to numpy array"""
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        if audio.ndim > 1:
            audio = audio.squeeze()
        return audio
    
    def compute_pesq(
        self,
        reference: Union[torch.Tensor, np.ndarray],
        degraded: Union[torch.Tensor, np.ndarray],
        mode: str = 'wb'
    ) -> float:
        """
        Compute PESQ (Perceptual Evaluation of Speech Quality)
        
        Args:
            reference: Clean reference audio
            degraded: Degraded/enhanced audio
            mode: 'wb' for wideband (16kHz) or 'nb' for narrowband (8kHz)
        
        Returns:
            PESQ score (higher is better, range: -0.5 to 4.5)
        """
        if not _PESQ_AVAILABLE:
            print("PESQ not available. Install 'pesq' to enable this metric.")
            return 0.0

        ref = self._to_numpy(reference)
        deg = self._to_numpy(degraded)
        
        # Ensure same length
        min_len = min(len(ref), len(deg))
        ref = ref[:min_len]
        deg = deg[:min_len]
        
        try:
            score = pesq(self.sample_rate, ref, deg, mode)
        except Exception as e:
            print(f"PESQ calculation failed: {e}")
            score = 0.0
        
        return score
    
    def compute_stoi(
        self,
        reference: Union[torch.Tensor, np.ndarray],
        degraded: Union[torch.Tensor, np.ndarray],
        extended: bool = False
    ) -> float:
        """
        Compute STOI (Short-Time Objective Intelligibility)
        
        Args:
            reference: Clean reference audio
            degraded: Degraded/enhanced audio
            extended: Use extended STOI (ESTOI)
        
        Returns:
            STOI score (higher is better, range: 0 to 1)
        """
        if not _STOI_AVAILABLE:
            print("STOI not available. Install 'pystoi' to enable this metric.")
            return 0.0

        ref = self._to_numpy(reference)
        deg = self._to_numpy(degraded)
        
        # Ensure same length
        min_len = min(len(ref), len(deg))
        ref = ref[:min_len]
        deg = deg[:min_len]
        
        try:
            score = stoi(ref, deg, self.sample_rate, extended=extended)
        except Exception as e:
            print(f"STOI calculation failed: {e}")
            score = 0.0
        
        return score
    
    def compute_si_snr(
        self,
        reference: Union[torch.Tensor, np.ndarray],
        degraded: Union[torch.Tensor, np.ndarray]
    ) -> float:
        """
        Compute SI-SNR (Scale-Invariant Signal-to-Noise Ratio)
        
        Args:
            reference: Clean reference audio
            degraded: Degraded/enhanced audio
        
        Returns:
            SI-SNR in dB (higher is better)
        """
        if isinstance(reference, np.ndarray):
            reference = torch.from_numpy(reference)
        if isinstance(degraded, np.ndarray):
            degraded = torch.from_numpy(degraded)
        
        # Ensure same length
        min_len = min(reference.shape[-1], degraded.shape[-1])
        reference = reference[..., :min_len]
        degraded = degraded[..., :min_len]
        
        # Zero-mean
        reference = reference - torch.mean(reference)
        degraded = degraded - torch.mean(degraded)
        
        # Compute projection
        s_target = torch.sum(reference * degraded) * reference / (torch.sum(reference ** 2) + 1e-8)
        e_noise = degraded - s_target
        
        # Compute SI-SNR
        si_snr = 10 * torch.log10(
            torch.sum(s_target ** 2) / (torch.sum(e_noise ** 2) + 1e-8)
        )
        
        return si_snr.item()
    
    def compute_sdr(
        self,
        reference: Union[torch.Tensor, np.ndarray],
        degraded: Union[torch.Tensor, np.ndarray]
    ) -> float:
        """
        Compute SDR (Signal-to-Distortion Ratio)
        
        Args:
            reference: Clean reference audio
            degraded: Degraded/enhanced audio
        
        Returns:
            SDR in dB (higher is better)
        """
        if isinstance(reference, np.ndarray):
            reference = torch.from_numpy(reference)
        if isinstance(degraded, np.ndarray):
            degraded = torch.from_numpy(degraded)
        
        # Ensure same length
        min_len = min(reference.shape[-1], degraded.shape[-1])
        reference = reference[..., :min_len]
        degraded = degraded[..., :min_len]
        
        # Compute distortion
        distortion = degraded - reference
        
        # Compute powers
        signal_power = torch.sum(reference ** 2)
        distortion_power = torch.sum(distortion ** 2)
        
        # Compute SDR
        if distortion_power == 0:
            return float('inf')
        
        sdr = 10 * torch.log10(signal_power / (distortion_power + 1e-8))
        
        return sdr.item()
    
    def compute_snr(
        self,
        clean: Union[torch.Tensor, np.ndarray],
        noisy: Union[torch.Tensor, np.ndarray]
    ) -> float:
        """
        Compute SNR (Signal-to-Noise Ratio)
        
        Args:
            clean: Clean signal
            noisy: Noisy signal
        
        Returns:
            SNR in dB (higher is better)
        """
        if isinstance(clean, np.ndarray):
            clean = torch.from_numpy(clean)
        if isinstance(noisy, np.ndarray):
            noisy = torch.from_numpy(noisy)
        
        # Ensure same length
        min_len = min(clean.shape[-1], noisy.shape[-1])
        clean = clean[..., :min_len]
        noisy = noisy[..., :min_len]
        
        # Compute noise
        noise = noisy - clean
        
        # Compute powers
        signal_power = torch.mean(clean ** 2)
        noise_power = torch.mean(noise ** 2)
        
        # Compute SNR
        if noise_power == 0:
            return float('inf')
        
        snr = 10 * torch.log10(signal_power / noise_power)
        
        return snr.item()
    
    def compute_all_metrics(
        self,
        reference: Union[torch.Tensor, np.ndarray],
        degraded: Union[torch.Tensor, np.ndarray]
    ) -> dict:
        """
        Compute all available metrics
        
        Args:
            reference: Clean reference audio
            degraded: Degraded/enhanced audio
        
        Returns:
            Dictionary with all metric scores
        """
        metrics = {}
        
        try:
            metrics['pesq'] = self.compute_pesq(reference, degraded)
        except:
            metrics['pesq'] = 0.0
        
        try:
            metrics['stoi'] = self.compute_stoi(reference, degraded)
        except:
            metrics['stoi'] = 0.0
        
        try:
            metrics['estoi'] = self.compute_stoi(reference, degraded, extended=True)
        except:
            metrics['estoi'] = 0.0
        
        try:
            metrics['si_snr'] = self.compute_si_snr(reference, degraded)
        except:
            metrics['si_snr'] = 0.0
        
        try:
            metrics['sdr'] = self.compute_sdr(reference, degraded)
        except:
            metrics['sdr'] = 0.0
        
        return metrics
    
    def print_metrics(self, metrics: dict):
        """Pretty print metrics"""
        print("\n" + "="*50)
        print("Audio Quality Metrics")
        print("="*50)
        for metric_name, value in metrics.items():
            print(f"{metric_name.upper():10s}: {value:7.4f}")
        print("="*50 + "\n")


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
