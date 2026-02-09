"""
Training script for Speech Enhancement Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
from pathlib import Path
import sys
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.transformer import TransformerSpeechEnhancement
from utils.dataset import create_dataloaders
from utils.audio_processing import AudioProcessor
from utils.metrics import MetricsCalculator, AverageMeter


class SpeechEnhancementTrainer:
    """Trainer for speech enhancement model"""
    
    def __init__(self, config_path: str, resume_from: str = None):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.resume_checkpoint = resume_from
        
        # Set device
        self.device = torch.device(
            self.config['hardware']['device'] 
            if torch.cuda.is_available() 
            else 'cpu'
        )
        print(f"Using device: {self.device}")
        
        # Set random seed
        torch.manual_seed(self.config['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config['seed'])
        
        # Initialize components
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        self.setup_loss()
        self.setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.best_metric = -float('inf')
        self.metrics_calculator = MetricsCalculator(
            sample_rate=self.config['model']['sample_rate']
        )
        
        # Resume from checkpoint if specified
        if self.resume_checkpoint:
            self.load_checkpoint(self.resume_checkpoint)
    
    def setup_model(self):
        """Initialize model"""
        model_config = self.config['model']
        self.model = TransformerSpeechEnhancement(
            n_fft=model_config['n_fft'],
            d_model=model_config['d_model'],
            nhead=model_config['nhead'],
            num_encoder_layers=model_config['num_encoder_layers'],
            dim_feedforward=model_config['dim_feedforward'],
            dropout=model_config['dropout'],
            use_cooperative=model_config['use_cooperative']
        ).to(self.device)
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def setup_data(self):
        """Setup data loaders"""
        data_config = self.config['data']
        
        self.audio_processor = AudioProcessor(
            sample_rate=data_config['sample_rate'],
            n_fft=self.config['model']['n_fft'],
            hop_length=self.config['model']['hop_length'],
            win_length=self.config['model']['win_length']
        )
        
        # Move audio processor to device
        self.audio_processor.to(self.device)
        
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            clean_dir=data_config['clean_dir'],
            noise_dir=data_config['noise_dir'],
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['hardware']['num_workers'],
            train_ratio=data_config['train_ratio'],
            val_ratio=data_config['val_ratio'],
            sample_rate=data_config['sample_rate'],
            segment_length=data_config['segment_length'],
            snr_range=data_config['augmentation']['noise_snr_range']
        )
        
        print(f"Data loaded: {len(self.train_loader)} train batches, "
              f"{len(self.val_loader)} val batches, {len(self.test_loader)} test batches")
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        train_config = self.config['training']
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            patience=train_config['scheduler_patience'],
            factor=train_config['scheduler_factor']
        )
    
    def setup_loss(self):
        """Setup loss function"""
        loss_type = self.config['training']['loss_type']
        
        if loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'mae':
            self.criterion = nn.L1Loss()
        elif loss_type == 'si_snr':
            self.criterion = self.si_snr_loss
        else:
            self.criterion = nn.MSELoss()
    
    def si_snr_loss(self, estimate, target):
        """Scale-Invariant SNR loss"""
        # Zero-mean
        estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)
        target = target - torch.mean(target, dim=-1, keepdim=True)
        
        # Compute projection
        s_target = torch.sum(target * estimate, dim=-1, keepdim=True) * target / \
                   (torch.sum(target ** 2, dim=-1, keepdim=True) + 1e-8)
        e_noise = estimate - s_target
        
        # Compute SI-SNR
        si_snr = 10 * torch.log10(
            torch.sum(s_target ** 2, dim=-1) / (torch.sum(e_noise ** 2, dim=-1) + 1e-8)
        )
        
        return -torch.mean(si_snr)  # Negative because we want to maximize
    
    def setup_logging(self):
        """Setup logging"""
        log_config = self.config['logging']
        
        if log_config['use_tensorboard']:
            log_dir = Path(log_config['log_dir'])
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        losses = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (noisy, clean) in enumerate(pbar):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            
            # Compute STFT
            noisy_stft = self.audio_processor.compute_stft(noisy)
            clean_stft = self.audio_processor.compute_stft(clean)
            
            # Get magnitudes
            noisy_mag = torch.abs(noisy_stft).transpose(1, 2)  # (B, T, F)
            clean_mag = torch.abs(clean_stft).transpose(1, 2)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_mask = self.model(noisy_mag)
            
            # Apply mask
            enhanced_mag = noisy_mag * pred_mask
            
            # Compute loss
            loss = self.criterion(enhanced_mag, clean_mag)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training']['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            losses.update(loss.item(), noisy.size(0))
            pbar.set_postfix({'loss': losses.avg})
            
            # Log to tensorboard
            if self.writer and batch_idx % self.config['logging']['log_interval'] == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/loss', loss.item(), step)
        
        return losses.avg
    
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        losses = AverageMeter()
        all_metrics = {metric: AverageMeter() for metric in ['pesq', 'stoi', 'si_snr']}
        
        with torch.no_grad():
            for noisy, clean in tqdm(self.val_loader, desc="Validating"):
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                
                # Compute STFT
                noisy_stft = self.audio_processor.compute_stft(noisy)
                clean_stft = self.audio_processor.compute_stft(clean)
                
                # Get magnitudes
                noisy_mag = torch.abs(noisy_stft).transpose(1, 2)
                clean_mag = torch.abs(clean_stft).transpose(1, 2)
                
                # Forward pass
                pred_mask = self.model(noisy_mag)
                enhanced_mag = noisy_mag * pred_mask
                
                # Compute loss
                loss = self.criterion(enhanced_mag, clean_mag)
                losses.update(loss.item(), noisy.size(0))
                
                # Reconstruct audio for metrics
                enhanced_stft = enhanced_mag.transpose(1, 2) * torch.exp(1j * torch.angle(noisy_stft))
                enhanced_audio = self.audio_processor.compute_istft(enhanced_stft)
                
                # Compute metrics (on first batch only for speed)
                if all_metrics['pesq'].count == 0:
                    for i in range(min(4, clean.size(0))):  # Compute on 4 samples
                        metrics = self.metrics_calculator.compute_all_metrics(
                            clean[i].cpu(),
                            enhanced_audio[i].cpu()
                        )
                        for key in all_metrics:
                            if key in metrics:
                                all_metrics[key].update(metrics[key])
        
        # Log metrics
        if self.writer:
            self.writer.add_scalar('val/loss', losses.avg, epoch)
            for metric_name, meter in all_metrics.items():
                if meter.count > 0:
                    self.writer.add_scalar(f'val/{metric_name}', meter.avg, epoch)
        
        print(f"Validation - Loss: {losses.avg:.4f}, "
              f"PESQ: {all_metrics['pesq'].avg:.4f}, "
              f"STOI: {all_metrics['stoi'].avg:.4f}")
        
        return losses.avg, all_metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }
        
        # Save regular checkpoint
        path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
        
        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
        
        # Save latest checkpoint for easy resuming
        latest_path = checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_metric = checkpoint['best_metric']
        
        print(f"Resumed from epoch {checkpoint['epoch']}, best metric: {self.best_metric:.4f}")
    
    def train(self):
        """Main training loop"""
        num_epochs = self.config['training']['num_epochs']
        patience = self.config['training']['patience']
        epochs_without_improvement = 0
        
        for epoch in range(self.current_epoch, num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_metrics = self.validate(epoch)
            
            # Update scheduler
            monitor_metric = val_metrics['pesq'].avg if val_metrics['pesq'].count > 0 else -val_loss
            self.scheduler.step(monitor_metric)
            
            # Check if best model
            is_best = monitor_metric > self.best_metric
            if is_best:
                self.best_metric = monitor_metric
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Save checkpoint
            if (epoch + 1) % self.config['checkpoint']['save_interval'] == 0:
                self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.config['training']['early_stopping']:
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        print("Training completed!")
        if self.writer:
            self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train Speech Enhancement Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from (e.g., checkpoints/latest_checkpoint.pt)')
    args = parser.parse_args()
    
    trainer = SpeechEnhancementTrainer(args.config, resume_from=args.resume)
    trainer.train()


if __name__ == '__main__':
    main()
