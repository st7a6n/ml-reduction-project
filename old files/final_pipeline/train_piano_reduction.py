# train_piano_reduction.py
"""
Complete training pipeline for piano reduction transformer
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import logging
import json
from tqdm import tqdm

# Import from your transformer.py
from transformer import (
    Config, MusicTokenizer, PianoReductionDataset, 
    PianoReductionTransformer, Trainer, collate_fn
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def verify_dataset(data_dir: str):
    """
    Verify that dataset is properly formatted
    """
    logger.info("="*80)
    logger.info("🔍 VERIFYING DATASET")
    logger.info("="*80)
    
    data_path = Path(data_dir)
    orch_dir = data_path / 'orchestral'
    piano_dir = data_path / 'piano'
    
    if not orch_dir.exists():
        logger.error(f"❌ Orchestral directory not found: {orch_dir}")
        return False
    
    if not piano_dir.exists():
        logger.error(f"❌ Piano directory not found: {piano_dir}")
        return False
    
    # Count files
    orch_files = list(orch_dir.glob('*.musicxml')) + list(orch_dir.glob('*.xml'))
    piano_files = list(piano_dir.glob('*.musicxml')) + list(piano_dir.glob('*.xml'))
    
    logger.info(f"Orchestral files: {len(orch_files)}")
    logger.info(f"Piano files:      {len(piano_files)}")
    
    # Find matching pairs
    orch_names = {f.stem for f in orch_files}
    piano_names = {f.stem for f in piano_files}
    
    matching = orch_names & piano_names
    orch_only = orch_names - piano_names
    piano_only = piano_names - orch_names
    
    logger.info("")
    logger.info(f"✓ Matching pairs:     {len(matching)}")
    
    if orch_only:
        logger.warning(f"⚠️  Orchestral only:    {len(orch_only)} files")
        logger.warning(f"   (first 5: {list(orch_only)[:5]})")
    
    if piano_only:
        logger.warning(f"⚠️  Piano only:         {len(piano_only)} files")
        logger.warning(f"   (first 5: {list(piano_only)[:5]})")
    
    logger.info("")
    
    if len(matching) == 0:
        logger.error("❌ No matching pairs found!")
        return False
    
    if len(matching) < 10:
        logger.warning(f"⚠️  Only {len(matching)} pairs - this is very little data!")
        logger.warning("   Model will likely overfit. Consider:")
        logger.warning("   1. Using smaller segments (4-8 measures)")
        logger.warning("   2. Adding data augmentation (transposition)")
        logger.warning("   3. Getting more training data")
    
    logger.info(f"✅ Dataset verified: {len(matching)} training pairs")
    logger.info("")
    
    return len(matching) > 0


def train_model(
    data_dir: str = "train_data_split",
    checkpoint_dir: str = "checkpoints",
    vocab_path: str = "vocab.json",
    num_epochs: int = 100,
    batch_size: int = 4,
    learning_rate: float = 0.0001,
    d_model: int = 256,
    num_layers: int = 4,
    max_seq_length: int = 2048
):
    """
    Complete training pipeline
    
    Args:
        data_dir: Directory with split training data
        checkpoint_dir: Where to save checkpoints
        vocab_path: Where to save vocabulary
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        d_model: Model dimension
        num_layers: Number of transformer layers
        max_seq_length: Maximum sequence length
    """
    
    logger.info("="*80)
    logger.info("🎵 PIANO REDUCTION TRANSFORMER - TRAINING")
    logger.info("="*80)
    logger.info("")
    
    # Verify dataset
    if not verify_dataset(data_dir):
        logger.error("Dataset verification failed!")
        return
    
    # Create configuration
    config = Config()
    config.data_dir = data_dir
    config.checkpoint_dir = checkpoint_dir
    config.vocab_path = vocab_path
    config.num_epochs = num_epochs
    config.batch_size = batch_size
    config.learning_rate = learning_rate
    config.d_model = d_model
    config.num_encoder_layers = num_layers
    config.num_decoder_layers = num_layers
    config.max_seq_length = max_seq_length
    
    logger.info("CONFIGURATION:")
    logger.info(f"  Device:           {config.device}")
    logger.info(f"  Data directory:   {config.data_dir}")
    logger.info(f"  Epochs:           {config.num_epochs}")
    logger.info(f"  Batch size:       {config.batch_size}")
    logger.info(f"  Learning rate:    {config.learning_rate}")
    logger.info(f"  Model dimension:  {config.d_model}")
    logger.info(f"  Transformer layers: {config.num_encoder_layers}")
    logger.info(f"  Max seq length:   {config.max_seq_length}")
    logger.info("")
    
    # Create tokenizer
    logger.info("📝 Creating tokenizer...")
    tokenizer = MusicTokenizer()
    
    # Load vocabulary if exists
    if Path(vocab_path).exists():
        logger.info(f"  Loading existing vocabulary from {vocab_path}")
        tokenizer.load_vocab(vocab_path)
    
    # Create dataset
    logger.info("📂 Loading dataset...")
    dataset = PianoReductionDataset(config.data_dir, tokenizer, config.max_seq_length)
    
    if len(dataset) == 0:
        logger.error("❌ No training data found!")
        return
    
    logger.info(f"  ✓ Loaded {len(dataset)} training examples")
    logger.info("")
    
    # Save vocabulary
    tokenizer.save_vocab(vocab_path)
    logger.info(f"✓ Vocabulary saved to {vocab_path}")
    
    # Update vocab size
    config.vocab_size = max(
        tokenizer.next_instrument_id,
        tokenizer.pitch_offset + 128,
        tokenizer.duration_offset + 100,
        tokenizer.velocity_offset + 128
    ) + 10
    
    logger.info(f"  Vocabulary size: {config.vocab_size}")
    logger.info("")
    
    # Split dataset
    train_size = int(0.85 * len(dataset))
    val_size = int(0.10 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    # Ensure at least 1 example in each split
    if val_size == 0:
        val_size = 1
        train_size -= 1
    if test_size == 0:
        test_size = 1
        train_size -= 1
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    logger.info("📊 Dataset split:")
    logger.info(f"  Training:   {len(train_dataset)} examples")
    logger.info(f"  Validation: {len(val_dataset)} examples")
    logger.info(f"  Test:       {len(test_dataset)} examples")
    logger.info("")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if config.device == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if config.device == "cuda" else False
    )
    
    # Create model
    logger.info("🤖 Creating model...")
    model = PianoReductionTransformer(config)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  ✓ Model created")
    logger.info(f"  Parameters: {num_params:,}")
    logger.info("")
    
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    logger.info("🏋️ Initializing trainer...")
    trainer = Trainer(model, train_loader, val_loader, config, tokenizer)
    logger.info("")
    
    # Training loop
    logger.info("="*80)
    logger.info("🚀 STARTING TRAINING")
    logger.info("="*80)
    logger.info("")
    
    try:
        trainer.train()
        
        logger.info("")
        logger.info("="*80)
        logger.info("✅ TRAINING COMPLETE!")
        logger.info("="*80)
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
        logger.info(f"Checkpoints saved to: {checkpoint_dir}")
        logger.info(f"Best model: {checkpoint_dir}/best_model.pt")
        logger.info("")
        
    except KeyboardInterrupt:
        logger.info("")
        logger.info("⚠️  Training interrupted by user")
        logger.info("Saving checkpoint...")
        
        checkpoint_path = Path(checkpoint_dir) / 'interrupted_checkpoint.pt'
        trainer.save_checkpoint(str(checkpoint_path))
        
        logger.info(f"✓ Checkpoint saved to: {checkpoint_path}")
        logger.info("You can resume training later by loading this checkpoint.")
        
    except Exception as e:
        logger.error("")
        logger.error("❌ Training failed with error:")
        logger.error(str(e))
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Piano Reduction Transformer")
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='train_data_split',
        help='Directory with split training data (default: train_data_split)'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='checkpoints',
        help='Directory to save checkpoints (default: checkpoints)'
    )
    parser.add_argument(
        '--vocab',
        type=str,
        default='vocab.json',
        help='Path to vocabulary file (default: vocab.json)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size (default: 4)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help='Learning rate (default: 0.0001)'
    )
    parser.add_argument(
        '--d_model',
        type=int,
        default=256,
        help='Model dimension (default: 256, smaller=faster but less capacity)'
    )
    parser.add_argument(
        '--layers',
        type=int,
        default=4,
        help='Number of transformer layers (default: 4)'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=2048,
        help='Maximum sequence length (default: 2048)'
    )
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        vocab_path=args.vocab,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        d_model=args.d_model,
        num_layers=args.layers,
        max_seq_length=args.max_length
    )