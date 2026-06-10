# split_training_data.py
"""
Split long paired scores into 8-measure training segments
"""

from music21 import converter
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def split_score_into_segments(
    input_file: str,
    output_dir: str,
    base_name: str,
    measures_per_segment: int = 8,
    overlap: int = 0
):
    """
    Split a score into segments
    
    Args:
        input_file: Input MusicXML file
        output_dir: Output directory
        base_name: Base name for output files
        measures_per_segment: Measures per segment
        overlap: Overlap between segments
    
    Returns:
        Number of segments created
    """
    logger.info(f"  Splitting {input_file}...")
    
    # Parse score
    score = converter.parse(input_file)
    
    # Get total measures
    if not score.parts:
        logger.error(f"    No parts found in {input_file}")
        return 0
    
    measures = score.parts[0].getElementsByClass('Measure')
    total_measures = len(measures)
    
    logger.info(f"    Total measures: {total_measures}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Split into segments
    segment_count = 0
    start_measure = 1
    
    while start_measure <= total_measures:
        end_measure = min(start_measure + measures_per_segment - 1, total_measures)
        
        # Skip if segment is too small
        if end_measure - start_measure + 1 < measures_per_segment // 2:
            logger.info(f"    Skipping incomplete segment {start_measure}-{end_measure}")
            break
        
        try:
            # Extract segment
            segment = score.measures(start_measure, end_measure)
            
            # Save segment
            output_file = output_path / f"{base_name}_m{start_measure:04d}-{end_measure:04d}.musicxml"
            segment.write('musicxml', fp=str(output_file))
            
            segment_count += 1
            
        except Exception as e:
            logger.warning(f"    Error extracting measures {start_measure}-{end_measure}: {e}")
        
        # Move to next segment
        start_measure += measures_per_segment - overlap
    
    logger.info(f"    Created {segment_count} segments")
    
    return segment_count


def split_all_pairs(
    input_dir: str = "train_data",
    output_dir: str = "train_data_split",
    measures_per_segment: int = 8,
    overlap: int = 0
):
    """
    Split all paired orchestral/piano scores
    
    Args:
        input_dir: Directory with 'symphony' and 'piano' subdirectories
        output_dir: Output directory for split data
        measures_per_segment: Measures per segment
        overlap: Overlap between segments
    """
    input_path = Path(input_dir)
    symphony_dir = input_path / 'symphony'
    piano_dir = input_path / 'piano'
    
    output_orch = Path(output_dir) / 'orchestral'
    output_piano = Path(output_dir) / 'piano'
    
    logger.info("="*80)
    logger.info("🎵 SPLITTING TRAINING DATA INTO SEGMENTS")
    logger.info("="*80)
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Segment size: {measures_per_segment} measures")
    logger.info(f"Overlap: {overlap} measures")
    logger.info("")
    
    total_segments = 0
    
    # Process each numbered pair
    for i in range(1, 10):  # Check pairs 1-9
        logger.info(f"{'─'*80}")
        logger.info(f"📁 PAIR {i}")
        logger.info(f"{'─'*80}")
        
        # Find files
        symphony_file = None
        piano_file = None
        
        for ext in ['.mxl', '.musicxml', '.xml']:
            sym_path = symphony_dir / f"{i}{ext}"
            piano_path = piano_dir / f"{i}{ext}"
            
            if sym_path.exists():
                symphony_file = sym_path
            if piano_path.exists():
                piano_file = piano_path
        
        if not symphony_file or not piano_file:
            logger.info(f"  Skipping pair {i} - files not found")
            continue
        
        logger.info(f"  Symphony: {symphony_file.name}")
        logger.info(f"  Piano:    {piano_file.name}")
        logger.info("")
        
        # Split symphony
        orch_segments = split_score_into_segments(
            str(symphony_file),
            str(output_orch),
            f"pair{i}",
            measures_per_segment,
            overlap
        )
        
        # Split piano
        piano_segments = split_score_into_segments(
            str(piano_file),
            str(output_piano),
            f"pair{i}",
            measures_per_segment,
            overlap
        )
        
        # Verify counts match
        if orch_segments != piano_segments:
            logger.warning(f"  ⚠️  Segment count mismatch: orch={orch_segments}, piano={piano_segments}")
        else:
            logger.info(f"  ✓ Created {orch_segments} matching segment pairs")
        
        total_segments += min(orch_segments, piano_segments)
        logger.info("")
    
    logger.info("="*80)
    logger.info("✅ SPLITTING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Total training examples created: {total_segments}")
    logger.info(f"Orchestral segments: {output_orch}")
    logger.info(f"Piano segments:      {output_piano}")
    logger.info("")
    
    return total_segments


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Split training data into segments")
    parser.add_argument(
        '--input_dir',
        type=str,
        default='train_data',
        help='Input directory with symphony/piano subdirectories'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='train_data_split',
        help='Output directory for split segments'
    )
    parser.add_argument(
        '--measures',
        type=int,
        default=8,
        help='Measures per segment (default: 8)'
    )
    parser.add_argument(
        '--overlap',
        type=int,
        default=0,
        help='Overlap between segments (default: 0)'
    )
    
    args = parser.parse_args()
    
    split_all_pairs(
        args.input_dir,
        args.output_dir,
        args.measures,
        args.overlap
    )