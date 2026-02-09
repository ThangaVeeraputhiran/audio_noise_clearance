# Dataset Download Guide

This guide explains how to download and prepare the datasets for your speech enhancement project.

## Quick Start - Download All Datasets

```bash
# Download both LibriSpeech and MUSAN (recommended)
python scripts/prepare_data.py --all

# This will download:
# - LibriSpeech train-clean-100 (~6.3 GB)
# - MUSAN dataset (~6 GB)
# Total: ~12.3 GB
```

## Individual Dataset Downloads

### 1. LibriSpeech (Clean Speech) - Required

```bash
# Download train-clean-100 (default, ~6.3 GB, 100 hours)
python scripts/prepare_data.py --librispeech

# Or choose a different subset:
python scripts/prepare_data.py --librispeech --subset dev-clean       # Small, for testing
python scripts/prepare_data.py --librispeech --subset train-clean-360 # Large, 360 hours
```

**Available subsets:**
- `dev-clean` - ~300 MB (for quick testing)
- `train-clean-100` - ~6.3 GB (100 hours, **recommended**)
- `train-clean-360` - ~23 GB (360 hours)
- `test-clean` - ~300 MB

### 2. MUSAN (Noise Sources) - Required

```bash
# Download MUSAN dataset (~6 GB)
python scripts/prepare_data.py --musan
```

**MUSAN includes:**
- Background noise (street, office, etc.)
- Background speech (babble)
- Background music

### 3. DEMAND (Optional - High-quality environmental noise)

DEMAND requires manual download due to Zenodo's authentication.

**Steps:**

1. Visit: https://zenodo.org/record/1227121

2. Download these files (16kHz, Channel 1):
   - `DKITCHEN_16k_1ch.zip`
   - `DLIVING_16k_1ch.zip`
   - `DWASHING_16k_1ch.zip`
   - `OMEETING_16k_1ch.zip`
   - `OOFFICE_16k_1ch.zip`
   - And any others you want

3. Place ZIP files in: `data/downloads/`

4. Run:
   ```bash
   python scripts/prepare_data.py --demand
   ```

## Dataset Organization

After download, your structure will be:

```
data/
├── clean/                          # Clean speech from LibriSpeech
│   ├── 1001_134707_000000.flac
│   ├── 1001_134707_000001.flac
│   └── ...
├── noise/                          # Noise from MUSAN
│   ├── noise/                      # Environmental noise
│   │   ├── noise_free-sound_*.wav
│   │   └── ...
│   ├── speech/                     # Background speech/babble
│   │   ├── speech_librivox_*.wav
│   │   └── ...
│   ├── music/                      # Background music
│   │   ├── music_fma_*.wav
│   │   └── ...
│   └── demand/                     # DEMAND (if downloaded)
│       ├── demand_DKITCHEN_*.wav
│       └── ...
├── noisy/                          # Auto-generated during training
└── enhanced/                       # Enhanced outputs
```

## Dataset Information

### LibriSpeech
- **Source:** OpenSLR
- **URL:** http://www.openslr.org/12
- **Format:** 16kHz FLAC
- **Content:** Audiobook readings (clean speech)
- **License:** CC BY 4.0

### MUSAN
- **Source:** OpenSLR
- **URL:** http://www.openslr.org/17
- **Format:** 16kHz WAV (will be converted)
- **Content:** Music, Speech, and Noise
- **Categories:**
  - Noise: Free sound clips (environmental sounds)
  - Speech: LibriVox audiobooks (background speech)
  - Music: FMA dataset excerpts
- **License:** Various (see dataset documentation)

### DEMAND (Optional)
- **Source:** Zenodo
- **URL:** https://zenodo.org/record/1227121
- **Format:** 16kHz WAV, Channel 1
- **Content:** Real-world recordings from:
  - Kitchen (DKITCHEN)
  - Living room (DLIVING)
  - Washing machine area (DWASHING)
  - Meeting room (OMEETING)
  - Office (OOFFICE)
  - Restaurant (PRESTO)
  - Cafeteria (PCAFETER)
  - And more...
- **License:** CC BY-SA 3.0

## Disk Space Requirements

| Dataset | Compressed | Extracted | Description |
|---------|-----------|-----------|-------------|
| LibriSpeech train-clean-100 | 6.3 GB | ~8 GB | 100 hours clean speech |
| LibriSpeech dev-clean | 300 MB | ~400 MB | Small test set |
| MUSAN | 6 GB | ~12 GB | All noise types |
| DEMAND (all) | ~2 GB | ~3 GB | Environmental noise |
| **Total (recommended)** | **~12 GB** | **~20 GB** | LibriSpeech-100 + MUSAN |

## Download Time Estimates

**On a 10 Mbps connection:**
- LibriSpeech train-clean-100: ~1.5 hours
- MUSAN: ~1.5 hours
- Total: ~3 hours

**On a 50 Mbps connection:**
- LibriSpeech train-clean-100: ~20 minutes
- MUSAN: ~20 minutes
- Total: ~40 minutes

## Troubleshooting

### Download interrupted?
The script will detect existing files and ask if you want to re-download.

### Out of disk space?
- Use smaller LibriSpeech subset: `--subset dev-clean`
- Delete archive files after extraction (script will prompt)
- Use external drive for data storage

### Slow download?
- Try different times of day
- Check your internet connection
- Consider downloading overnight

### Dataset not extracting?
- Ensure you have enough disk space (2x compressed size)
- Check that tar/gzip is installed
- Try re-downloading the archive

## After Download

1. **Verify downloads:**
   ```bash
   python scripts/prepare_data.py --create_readmes
   ```

2. **Check dataset summary:**
   The script automatically shows a summary after download.

3. **Start training:**
   ```bash
   python scripts/train.py --config configs/config.yaml
   ```

## Advanced Options

```bash
# Download everything at once
python scripts/prepare_data.py --all

# Download only LibriSpeech with specific subset
python scripts/prepare_data.py --librispeech --subset train-clean-360

# Download only MUSAN
python scripts/prepare_data.py --musan

# Create/update README files in directories
python scripts/prepare_data.py --create_readmes

# Custom data directory
python scripts/prepare_data.py --all --data_dir /path/to/custom/location
```

## Storage Tips

1. **Remove archives after extraction** - Save ~12 GB
2. **Use dev-clean for testing** - Only 300 MB
3. **External storage** - Move data folder to external drive
4. **Cloud storage** - Upload to cloud after download

## Citation

If you use these datasets in your project, please cite:

**LibriSpeech:**
```
@inproceedings{panayotov2015librispeech,
  title={Librispeech: an ASR corpus based on public domain audio books},
  author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle={2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={5206--5210},
  year={2015},
  organization={IEEE}
}
```

**MUSAN:**
```
@article{snyder2015musan,
  title={MUSAN: A Music, Speech, and Noise Corpus},
  author={Snyder, David and Chen, Guoguo and Povey, Daniel},
  journal={arXiv preprint arXiv:1510.08484},
  year={2015}
}
```

## Need Help?

- Check README.md for project overview
- See GETTING_STARTED.md for setup instructions
- Refer to configs/config.yaml for configuration options
