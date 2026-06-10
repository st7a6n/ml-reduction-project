# resume_training.py
"""
Resume training from a checkpoint
"""

import warnings
warnings.filterwarnings('ignore')

import torch
from pathlib import Path
from transformer import (
    Config, MusicTokenizer, PianoReductionDataset,
    PianoReductionTransformer, Trainer, collate_fn
)
from torch.utils.data import DataLoader, random_split

def resume_training(
    checkpoint_path: str = "checkpoints/interrupted_checkpoint.pt",
    additional_epochs: int = 50,
    data_dir: str = "train_data_split"
):
    """
    Resume training from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint to resume from
        additional_epochs: How many more epochs to train
        data_dir: Training data directory
    """
    
    print("="*80)
    print("🔄 RESUMING TRAINING")
    print("="*80)
    print()
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    config = checkpoint['config']
    current_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    
    print(f"  ✓ Checkpoint loaded")
    print(f"  Current epoch: {current_epoch}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print()
    
    # Update epochs
    config.num_epochs = current_epoch + additional_epochs
    print(f"Will train from epoch {current_epoch} to epoch {config.num_epochs}")
    print()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = MusicTokenizer()
    tokenizer.load_vocab(config.vocab_path)
    print(f"  ✓ Vocabulary loaded")
    print()
    
    # Load dataset
    print("Loading dataset...")
    dataset = PianoReductionDataset(data_dir, tokenizer, config.max_seq_length)
    print(f"  ✓ {len(dataset)} examples")
    print()
    
    # Split dataset (use same split as before)
    train_size = int(0.85 * len(dataset))
    val_size = int(0.10 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    if val_size == 0:
        val_size = 1
        train_size -= 1
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(config.device == "cuda")
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(config.device == "cuda")
    )
    
    # Create model
    print("Rebuilding model...")
    model = PianoReductionTransformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✓ Model restored")
    print()
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config, tokenizer)
    
    # Restore trainer state
    trainer.current_epoch = current_epoch
    trainer.global_step = checkpoint['global_step']
    trainer.best_val_loss = best_val_loss
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print("="*80)
    print("🚀 RESUMING TRAINING")
    print("="*80)
    print()
    
    # Continue training
    try:
        trainer.train()
        
        print()
        print("="*80)
        print("✅ TRAINING COMPLETE")
        print("="*80)
        print(f"Final best val loss: {trainer.best_val_loss:.4f}")
        print()
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted again")
        checkpoint_path = Path(config.checkpoint_dir) / 'interrupted_checkpoint.pt'
        trainer.save_checkpoint(str(checkpoint_path))
        print(f"Checkpoint saved: {checkpoint_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Resume training from checkpoint")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/interrupted_checkpoint.pt',
        help='Checkpoint to resume from'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Additional epochs to train'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='train_data_split',
        help='Training data directory'
    )
    
    args = parser.parse_args()
    
    resume_training(args.checkpoint, args.epochs, args.data_dir)