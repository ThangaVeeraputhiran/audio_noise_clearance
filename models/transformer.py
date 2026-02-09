"""
Transformer-based Speech Enhancement Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerSpeechEnhancement(nn.Module):
    """
    Transformer-based model for speech enhancement
    Implements architecture from the IEEE Access paper
    """
    
    def __init__(
        self,
        n_fft: int = 1024,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_cooperative: bool = True
    ):
        super().__init__()
        
        self.n_fft = n_fft
        self.d_model = d_model
        self.use_cooperative = use_cooperative
        
        # Input projection
        self.input_proj = nn.Linear(n_fft // 2 + 1, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Output projection for magnitude mask
        self.magnitude_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, n_fft // 2 + 1),
            nn.Sigmoid()  # Mask values between 0 and 1
        )
        
        # Cooperative learning: auxiliary task head
        if use_cooperative:
            self.auxiliary_head = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, n_fft // 2 + 1),
                nn.Sigmoid()
            )
    
    def forward(self, noisy_magnitude, return_auxiliary=False):
        """
        Forward pass
        
        Args:
            noisy_magnitude: (batch_size, time_frames, freq_bins)
            return_auxiliary: Whether to return auxiliary task output
        
        Returns:
            magnitude_mask: (batch_size, time_frames, freq_bins)
            auxiliary_output: (optional) auxiliary task output
        """
        # Project input to model dimension
        x = self.input_proj(noisy_magnitude)  # (B, T, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (B, T, d_model)
        
        # Generate magnitude mask
        magnitude_mask = self.magnitude_head(x)  # (B, T, freq_bins)
        
        # Auxiliary task (if cooperative learning)
        if self.use_cooperative and return_auxiliary:
            auxiliary_output = self.auxiliary_head(x)
            return magnitude_mask, auxiliary_output
        
        return magnitude_mask
    
    def enhance(self, noisy_stft):
        """
        Enhance noisy audio using trained model
        
        Args:
            noisy_stft: Complex STFT of noisy audio (B, freq_bins, time_frames)
        
        Returns:
            enhanced_stft: Enhanced complex STFT
        """
        # Get magnitude and phase
        noisy_magnitude = torch.abs(noisy_stft)  # (B, F, T)
        noisy_phase = torch.angle(noisy_stft)
        
        # Transpose for model input
        noisy_magnitude = noisy_magnitude.transpose(1, 2)  # (B, T, F)
        
        # Predict mask
        with torch.no_grad():
            magnitude_mask = self.forward(noisy_magnitude)
        
        # Apply mask to magnitude
        enhanced_magnitude = noisy_magnitude * magnitude_mask
        
        # Transpose back
        enhanced_magnitude = enhanced_magnitude.transpose(1, 2)  # (B, F, T)
        
        # Reconstruct complex STFT with original phase
        enhanced_stft = enhanced_magnitude * torch.exp(1j * noisy_phase)
        
        return enhanced_stft


class ComplexTransformerSpeechEnhancement(nn.Module):
    """
    Complex-valued Transformer for speech enhancement
    Processes both magnitude and phase information
    """
    
    def __init__(
        self,
        n_fft: int = 1024,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_fft = n_fft
        self.d_model = d_model
        
        # Separate processing for real and imaginary parts
        self.real_input_proj = nn.Linear(n_fft // 2 + 1, d_model // 2)
        self.imag_input_proj = nn.Linear(n_fft // 2 + 1, d_model // 2)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Output projections
        self.real_output_proj = nn.Linear(d_model, n_fft // 2 + 1)
        self.imag_output_proj = nn.Linear(d_model, n_fft // 2 + 1)
    
    def forward(self, noisy_stft):
        """
        Forward pass with complex STFT input
        
        Args:
            noisy_stft: Complex STFT (B, F, T)
        
        Returns:
            enhanced_stft: Enhanced complex STFT (B, F, T)
        """
        # Separate real and imaginary parts
        real_part = noisy_stft.real.transpose(1, 2)  # (B, T, F)
        imag_part = noisy_stft.imag.transpose(1, 2)  # (B, T, F)
        
        # Project to model dimension
        real_encoded = self.real_input_proj(real_part)  # (B, T, d_model/2)
        imag_encoded = self.imag_input_proj(imag_part)  # (B, T, d_model/2)
        
        # Concatenate
        x = torch.cat([real_encoded, imag_encoded], dim=-1)  # (B, T, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Predict real and imaginary parts
        enhanced_real = self.real_output_proj(x).transpose(1, 2)  # (B, F, T)
        enhanced_imag = self.imag_output_proj(x).transpose(1, 2)  # (B, F, T)
        
        # Reconstruct complex STFT
        enhanced_stft = torch.complex(enhanced_real, enhanced_imag)
        
        return enhanced_stft
