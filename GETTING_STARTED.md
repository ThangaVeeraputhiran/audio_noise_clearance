# Speech Enhancement Project - Getting Started

This guide will help you set up and run the speech enhancement project.

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Create README files in data directories
python scripts/prepare_data.py --create_readmes

# Optional: Download sample dataset
python scripts/prepare_data.py --download_librispeech
```

#### Manual Data Preparation

**Clean Speech:**
- Place clean speech files in `data/clean/`
- Recommended: LibriSpeech, VCTK, Common Voice
- Formats: .wav, .flac, .mp3

**Noise:**
- Place noise files in `data/noise/`
- Recommended: MS-SNSD, DEMAND, UrbanSound8K
- Types: white, babble, street, office, cafe

### 3. Training

```bash
# Train with default configuration
python scripts/train.py --config configs/config.yaml

# Monitor training with TensorBoard
tensorboard --logdir results/logs
```

### 4. Inference

```bash
# Enhance a single file
python scripts/inference.py \
    --checkpoint checkpoints/best_model.pt \
    --input path/to/noisy.wav \
    --output path/to/enhanced.wav

# Enhance all files in a directory
python scripts/inference.py \
    --checkpoint checkpoints/best_model.pt \
    --input data/noisy/ \
    --output data/enhanced/

# With reference for metrics
python scripts/inference.py \
    --checkpoint checkpoints/best_model.pt \
    --input noisy.wav \
    --output enhanced.wav \
    --reference clean.wav
```

## Project Structure Explained

```
Noise_cancellation_project/
├── configs/              # Configuration files
│   └── config.yaml      # Main training config
├── data/                # Data directories
│   ├── clean/          # Clean speech samples
│   ├── noise/          # Noise samples
│   ├── noisy/          # Mixed noisy speech
│   └── enhanced/       # Model outputs
├── models/             # Model architectures
│   ├── transformer.py  # Transformer model
│   └── attention.py    # Attention mechanisms
├── utils/              # Utility functions
│   ├── audio_processing.py  # Audio I/O and processing
│   ├── metrics.py      # Evaluation metrics
│   └── dataset.py      # Dataset classes
├── scripts/            # Training and inference scripts
│   ├── train.py       # Training script
│   ├── inference.py   # Inference script
│   └── prepare_data.py # Data preparation
└── results/           # Training outputs
    ├── logs/          # TensorBoard logs
    └── checkpoints/   # Model checkpoints
```

## Configuration

Edit `configs/config.yaml` to customize:

- **Model**: Architecture parameters (d_model, nhead, layers)
- **Training**: Batch size, learning rate, epochs
- **Data**: Paths, augmentation, SNR range
- **Evaluation**: Metrics to compute

## Evaluation Metrics

The model is evaluated using:
- **PESQ**: Perceptual quality (range: -0.5 to 4.5, higher better)
- **STOI**: Speech intelligibility (range: 0 to 1, higher better)
- **SI-SNR**: Signal quality (dB, higher better)
- **SDR**: Distortion ratio (dB, higher better)

## Tips for Better Results

1. **Data Quality**: Use high-quality clean speech (16kHz+)
2. **Diverse Noise**: Include various noise types and levels
3. **Training Time**: Train for 50-100 epochs minimum
4. **Hyperparameters**: Experiment with d_model, nhead, layers
5. **Augmentation**: Enable data augmentation for robustness

## Common Issues

### CUDA Out of Memory
- Reduce `batch_size` in config.yaml
- Reduce `segment_length` in config.yaml
- Use gradient accumulation

### Poor Performance
- Check data quality and alignment
- Increase model capacity (d_model, layers)
- Train for more epochs
- Adjust learning rate

### Slow Training
- Reduce `num_workers` if using slow storage
- Use smaller `segment_length`
- Enable mixed precision training

## Next Steps

1. ✅ Set up environment
2. ✅ Prepare dataset
3. ⏳ Train initial model
4. ⏳ Evaluate performance
5. ⏳ Fine-tune hyperparameters
6. ⏳ Test on real-world data
7. ⏳ Write project report

## Resources

- Paper: "Enhancing Model Robustness in Noisy Environments" (IEEE Access 2025)
- PyTorch: https://pytorch.org/docs/
- TorchAudio: https://pytorch.org/audio/
- PESQ: https://github.com/ludlows/python-pesq
- STOI: https://github.com/mpariente/pystoi

## Contact

For questions or issues, please refer to the main README.md or create an issue.

Good luck with your project! 🎯
