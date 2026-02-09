# Training Guide

## How to Pause and Resume Training

### Starting Fresh Training
```bash
python scripts/train.py --config configs/config.yaml
```

### Pausing Training
To pause training at any time:
- **Press `Ctrl+C`** in the terminal
- The training will stop and the latest checkpoint will be saved automatically

### Resuming Training
To resume from where you left off:
```bash
python scripts/train.py --config configs/config.yaml --resume checkpoints/latest_checkpoint.pt
```

Or resume from a specific epoch:
```bash
python scripts/train.py --config configs/config.yaml --resume checkpoints/checkpoint_epoch_10.pt
```

### Checkpoint Locations
Checkpoints are automatically saved in the `checkpoints/` directory:
- **`latest_checkpoint.pt`** - Most recent checkpoint (updated every 5 epochs)
- **`best_model.pt`** - Best performing model based on validation metrics
- **`checkpoint_epoch_X.pt`** - Checkpoint from specific epoch X

### Automatic Checkpointing
- Checkpoints are saved every **5 epochs** (configurable in config.yaml)
- Training state includes:
  - Model weights
  - Optimizer state
  - Learning rate scheduler state
  - Current epoch number
  - Best validation metric

### Monitoring Training Progress
To monitor training in real-time with TensorBoard:
```bash
tensorboard --logdir results/logs
```
Then open browser to `http://localhost:6006`

### Tips
1. **Long training sessions**: Training 100 epochs on CPU may take days. Feel free to pause overnight.
2. **Check progress**: Look at `checkpoints/` folder to see saved checkpoints
3. **GPU recommended**: If you have a GPU, edit `configs/config.yaml` and change `device: "cpu"` to `device: "cuda"`
4. **Reduce epochs for testing**: Edit `num_epochs` in config.yaml to test the pipeline first

### Example Workflow
```bash
# Start training
python scripts/train.py --config configs/config.yaml

# ... training runs for a while ...
# Press Ctrl+C to pause

# Check latest checkpoint
ls checkpoints/

# Resume later
python scripts/train.py --config configs/config.yaml --resume checkpoints/latest_checkpoint.pt
```

### Training Status
- **Current Setup**: 100 epochs, ~10 hours per epoch on CPU
- **Total Time**: ~40 days on CPU (consider GPU or reducing epochs)
- **Batch Size**: 16 (configurable)
- **Checkpoint Frequency**: Every 5 epochs
