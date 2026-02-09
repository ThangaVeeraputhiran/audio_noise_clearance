"""
Data preparation script
Downloads and prepares datasets for speech enhancement
Includes: LibriSpeech, MUSAN, and DEMAND datasets
"""

import argparse
from pathlib import Path
import urllib.request
import zipfile
import tarfile
import shutil
from tqdm import tqdm
import os
import time


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: str, max_retries: int = 5, timeout: int = 30):
    """Download file with progress bar, resume support, and retries."""
    print(f"\nDownloading: {url}")
    print(f"Saving to: {output_path}")

    output_path = Path(output_path)

    for attempt in range(1, max_retries + 1):
        try:
            # Check if file exists and get its size
            resume_header = {}
            if output_path.exists():
                existing_size = output_path.stat().st_size
                print(f"Resuming download from {existing_size} bytes...")
                resume_header = {'Range': f'bytes={existing_size}-'}

            req = urllib.request.Request(url, headers=resume_header)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                total_size = int(response.headers.get('content-length', 0))
                if output_path.exists():
                    total_size += output_path.stat().st_size

                with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, total=total_size,
                                         desc=url.split('/')[-1]) as t:
                    with open(output_path, 'ab') as f:
                        while True:
                            chunk = response.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
                            t.update(len(chunk))

            if total_size > 0:
                actual_size = output_path.stat().st_size
                if actual_size < total_size:
                    raise IOError(
                        f"Download incomplete: got {actual_size} of {total_size} bytes"
                    )

            print(f"Download complete: {output_path}")
            return
        except urllib.error.HTTPError as e:
            if e.code == 416:
                print("File already complete!")
                return
            raise
        except (urllib.error.URLError, IOError, OSError) as e:
            if attempt == max_retries:
                raise
            wait_s = 5 * attempt
            if url.startswith("https://www.openslr.org/"):
                url = url.replace("https://", "http://", 1)
            print(f"Network error: {e}. Retrying in {wait_s}s (attempt {attempt}/{max_retries})...")
            time.sleep(wait_s)


def prepare_librispeech(output_dir: Path, subset: str = "train-clean-100"):
    """
    Download LibriSpeech dataset for clean speech
    
    Args:
        output_dir: Base output directory
        subset: Dataset subset (train-clean-100, dev-clean, etc.)
    """
    print("\n" + "="*70)
    print(f"DOWNLOADING LIBRISPEECH: {subset}")
    print("="*70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset URLs
    urls = {
        "train-clean-100": "http://www.openslr.org/resources/12/train-clean-100.tar.gz",
        "train-clean-360": "http://www.openslr.org/resources/12/train-clean-360.tar.gz",
        "dev-clean": "http://www.openslr.org/resources/12/dev-clean.tar.gz",
        "test-clean": "http://www.openslr.org/resources/12/test-clean.tar.gz"
    }
    
    if subset not in urls:
        print(f"Error: Unknown subset '{subset}'")
        print(f"Available: {list(urls.keys())}")
        return
    
    url = urls[subset]
    tar_path = output_dir / f"{subset}.tar.gz"
    
    # Download if not exists (resume if partial)
    if tar_path.exists():
        print(f"Archive exists: {tar_path}")
        print(f"Size: {tar_path.stat().st_size / (1024**3):.2f} GB")
    
    download_file(url, str(tar_path))
    
    # Extract
    print("\nExtracting LibriSpeech files...")
    extract_dir = output_dir / "temp_librispeech"
    extract_dir.mkdir(exist_ok=True)
    
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(extract_dir)
    
    # Organize files into clean directory
    clean_dir = output_dir / "clean"
    clean_dir.mkdir(exist_ok=True)
    
    print(f"\nOrganizing clean speech files...")
    librispeech_dir = extract_dir / "LibriSpeech" / subset
    
    file_count = 0
    if librispeech_dir.exists():
        for speaker_dir in tqdm(list(librispeech_dir.iterdir()), desc="Processing speakers"):
            if speaker_dir.is_dir():
                for chapter_dir in speaker_dir.iterdir():
                    if chapter_dir.is_dir():
                        for audio_file in chapter_dir.glob("*.flac"):
                            # Create unique filename with speaker and chapter info
                            new_name = f"{speaker_dir.name}_{chapter_dir.name}_{audio_file.name}"
                            shutil.copy(audio_file, clean_dir / new_name)
                            file_count += 1
    
    print(f"\n✓ Processed {file_count} clean speech files")
    print(f"✓ Clean speech saved to: {clean_dir}")
    
    # Cleanup
    print("\nCleaning up temporary files...")
    shutil.rmtree(extract_dir, ignore_errors=True)
    
    # Optionally remove archive to save space
    keep_archive = input("\nKeep downloaded archive? (y/n): ").lower()
    if keep_archive != 'y':
        tar_path.unlink()
        print("Archive removed.")


def prepare_musan(output_dir: Path):
    """
    Download MUSAN dataset (Music, Speech, and Noise)
    Contains various noise types including background speech
    """
    print("\n" + "="*70)
    print("DOWNLOADING MUSAN DATASET")
    print("="*70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    url = "https://www.openslr.org/resources/17/musan.tar.gz"
    tar_path = output_dir / "musan.tar.gz"
    
    # Download if not exists (resume if partial)
    if tar_path.exists():
        print(f"Archive exists: {tar_path}")
        print(f"Size: {tar_path.stat().st_size / (1024**3):.2f} GB")
    
    # Download + extract (retry if archive corrupted)
    extract_dir = output_dir / "temp_musan"
    for attempt in range(1, 4):
        download_file(url, str(tar_path))

        # Extract
        print("\nExtracting MUSAN files...")
        extract_dir.mkdir(exist_ok=True)

        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
            break
        except (tarfile.ReadError, EOFError) as e:
            print(f"Archive appears corrupted ({e}). Retrying download...")
            if tar_path.exists():
                tar_path.unlink()
            shutil.rmtree(extract_dir, ignore_errors=True)
            if attempt == 3:
                raise
    
    # Organize files into noise directory
    noise_dir = output_dir / "noise"
    noise_dir.mkdir(exist_ok=True)
    
    print(f"\nOrganizing noise files...")
    musan_dir = extract_dir / "musan"
    
    file_count = 0
    if musan_dir.exists():
        # Process each category: music, speech, noise
        for category in ['noise', 'speech', 'music']:
            category_dir = musan_dir / category
            if category_dir.exists():
                category_output = noise_dir / category
                category_output.mkdir(exist_ok=True)
                
                for audio_file in tqdm(list(category_dir.rglob("*.wav")), 
                                      desc=f"Processing {category}"):
                    # Create organized filename
                    rel_path = audio_file.relative_to(category_dir)
                    new_name = f"{category}_{str(rel_path).replace(os.sep, '_')}"
                    shutil.copy(audio_file, category_output / new_name)
                    file_count += 1
    
    print(f"\n✓ Processed {file_count} noise files")
    print(f"✓ Noise files saved to: {noise_dir}")
    print(f"  - Background noise: {noise_dir / 'noise'}")
    print(f"  - Background speech: {noise_dir / 'speech'}")
    print(f"  - Background music: {noise_dir / 'music'}")
    
    # Cleanup
    print("\nCleaning up temporary files...")
    shutil.rmtree(extract_dir, ignore_errors=True)
    
    # Optionally remove archive
    keep_archive = input("\nKeep downloaded archive? (y/n): ").lower()
    if keep_archive != 'y':
        tar_path.unlink()
        print("Archive removed.")


def prepare_demand(output_dir: Path):
    """
    Download DEMAND dataset (Diverse Environments Multichannel Acoustic Noise)
    High-quality environmental noise recordings
    """
    print("\n" + "="*70)
    print("DOWNLOADING DEMAND DATASET")
    print("="*70)
    print("\nNote: DEMAND requires manual download from Zenodo")
    print("URL: https://zenodo.org/record/1227121")
    print("\nInstructions:")
    print("1. Visit the URL above")
    print("2. Download: DKITCHEN_16k_1ch.zip, DLIVING_16k_1ch.zip, etc.")
    print("3. Place ZIP files in the 'downloads' folder")
    print("4. Re-run this script with --demand flag")
    
    downloads_dir = output_dir / "downloads"
    downloads_dir.mkdir(exist_ok=True)
    
    print(f"\nPlace DEMAND .zip files in: {downloads_dir}")
    
    # Check if any DEMAND files exist
    demand_zips = list(downloads_dir.glob("D*.zip"))
    
    if not demand_zips:
        print("\nNo DEMAND files found. Download manually and re-run.")
        return
    
    print(f"\nFound {len(demand_zips)} DEMAND archive(s)")
    
    # Extract and organize
    noise_dir = output_dir / "noise" / "demand"
    noise_dir.mkdir(parents=True, exist_ok=True)
    
    file_count = 0
    for zip_file in demand_zips:
        print(f"\nExtracting: {zip_file.name}")
        
        extract_dir = output_dir / "temp_demand"
        extract_dir.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Copy WAV files
        for wav_file in extract_dir.rglob("*.wav"):
            # Use channel 1 as specified
            if "ch01" in wav_file.name or "_1." in wav_file.name:
                env_name = zip_file.stem.replace("_16k_1ch", "")
                new_name = f"demand_{env_name}_{wav_file.name}"
                shutil.copy(wav_file, noise_dir / new_name)
                file_count += 1
        
        # Cleanup temp
        shutil.rmtree(extract_dir, ignore_errors=True)
    
    print(f"\n✓ Processed {file_count} DEMAND noise files")
    print(f"✓ DEMAND noise saved to: {noise_dir}")


def print_dataset_summary(data_dir: Path):
    """Print summary of downloaded datasets"""
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    
    # Count clean files
    clean_dir = data_dir / "clean"
    clean_count = len(list(clean_dir.glob("*.flac"))) if clean_dir.exists() else 0
    
    # Count noise files
    noise_dir = data_dir / "noise"
    noise_count = 0
    noise_types = []
    if noise_dir.exists():
        for subdir in noise_dir.iterdir():
            if subdir.is_dir():
                count = len(list(subdir.rglob("*.wav")))
                noise_count += count
                noise_types.append(f"{subdir.name}: {count} files")
        # Also count direct files
        noise_count += len(list(noise_dir.glob("*.wav")))
    
    print(f"\nClean Speech Files: {clean_count}")
    print(f"Location: {clean_dir}")
    
    print(f"\nNoise Files: {noise_count}")
    print(f"Location: {noise_dir}")
    for noise_type in noise_types:
        print(f"  - {noise_type}")
    
    # Estimate dataset size
    total_duration_hours = clean_count * 10 / 3600  # Rough estimate
    print(f"\nEstimated clean speech duration: ~{total_duration_hours:.1f} hours")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Download and prepare datasets for speech enhancement',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets
  python scripts/prepare_data.py --all
  
  # Download specific datasets
  python scripts/prepare_data.py --librispeech --musan
  
  # Download LibriSpeech with specific subset
  python scripts/prepare_data.py --librispeech --subset train-clean-100
        """
    )
    
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Base data directory (default: data)')
    parser.add_argument('--all', action='store_true',
                       help='Download all datasets (LibriSpeech + MUSAN)')
    parser.add_argument('--librispeech', action='store_true',
                       help='Download LibriSpeech clean speech dataset')
    parser.add_argument('--subset', type=str, default='train-clean-100',
                       choices=['train-clean-100', 'train-clean-360', 'dev-clean', 'test-clean'],
                       help='LibriSpeech subset to download (default: train-clean-100)')
    parser.add_argument('--musan', action='store_true',
                       help='Download MUSAN noise dataset')
    parser.add_argument('--demand', action='store_true',
                       help='Prepare DEMAND dataset (requires manual download)')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("SPEECH ENHANCEMENT DATA PREPARATION")
    print("="*70)
    
    # Download datasets
    if args.all or args.librispeech:
        prepare_librispeech(data_dir, subset=args.subset)
    
    if args.all or args.musan:
        prepare_musan(data_dir)
    
    if args.demand:
        prepare_demand(data_dir)
    
    # Show summary
    if args.librispeech or args.musan or args.all:
        print_dataset_summary(data_dir)
    
    # Next steps
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    
    if not (args.all or args.librispeech or args.musan or args.demand):
        print("\nNo datasets selected for download.")
        print("\nQuick start:")
        print("  python scripts/prepare_data.py --all")
        print("\nFor more options:")
        print("  python scripts/prepare_data.py --help")
    else:
        clean_dir = data_dir / 'clean'
        noise_dir = data_dir / 'noise'
        
        clean_exists = clean_dir.exists() and any(clean_dir.glob("*.flac"))
        noise_exists = noise_dir.exists() and (
            any(noise_dir.glob("*.wav")) or 
            any(noise_dir.rglob("*.wav"))
        )
        
        if clean_exists and noise_exists:
            print("\n✓ Datasets ready for training!")
            print("\nTo start training:")
            print("  python scripts/train.py --config configs/config.yaml")
            print("\nTo monitor training:")
            print("  tensorboard --logdir results/logs")
        else:
            if not clean_exists:
                print(f"\n⚠ No clean speech files found in: {clean_dir}")
                print("  Run: python scripts/prepare_data.py --librispeech")
            if not noise_exists:
                print(f"\n⚠ No noise files found in: {noise_dir}")
                print("  Run: python scripts/prepare_data.py --musan")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
