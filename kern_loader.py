"""
kern_loader.py
==============
Utilities for loading the local CCARH Beethoven Symphony No. 5 Kern/Humdrum
parts and combining each movement into a music21 Score.

Examples:
    python kern_loader.py --inspect
    python kern_loader.py --movement 1 --write-xml beethoven5_mvt1.musicxml
"""

from __future__ import annotations

import argparse
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from music21 import converter, interval, stream


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "symph5data"
CCARH_SYMPH5_URL = "https://kern.ccarh.org/cgi-bin/ksdata?l=users/craig/classical/beethoven/symphonies/symphony5&format=zip"


def download_beethoven_symphonies_kern(
    destination: str = str(DEFAULT_DATA_DIR),
    url: str = CCARH_SYMPH5_URL,
) -> Path:
    """
    Fetch Beethoven Symphony No. 5 Kern data from CCARH as a zip archive.

    The project already includes `symph5data`; use this only to refresh data.
    """
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    archive_path = dest.parent / "beethoven_symphony5_kern.zip"
    print(f"Downloading CCARH data to {archive_path}")
    urllib.request.urlretrieve(url, archive_path)
    shutil.unpack_archive(str(archive_path), str(dest))
    return dest


def _movement_dir(data_dir: Path, movement_num: int) -> Path:
    candidates = [
        data_dir / f"{movement_num:02d}",
        data_dir / f"movement{movement_num}",
        data_dir / str(movement_num),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Movement {movement_num} not found under {data_dir}")


def _part_files(movement_path: Path) -> List[Path]:
    files = []
    for path in sorted(movement_path.iterdir()):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.startswith(("old", "mchan")) or name in {"temp", ".ds_store"}:
            continue
        if name.startswith(("s", "p")) or path.suffix == ".krn":
            files.append(path)
    return files


def load_kern_file(path: str) -> stream.Stream:
    """
    Parse one local Beethoven part file with music21.

    The folder is historically described as Kern/Humdrum, but these CCARH
    files are SCORE/MuseData-style text exports. Try MuseData first, then
    Humdrum as a fallback for genuinely Kern-formatted replacements.
    """
    errors = []
    for fmt in ("musedata", "humdrum"):
        try:
            return converter.parse(path, format=fmt)
        except Exception as exc:
            errors.append(f"{fmt}: {exc}")
    raise RuntimeError("; ".join(errors))


def infer_transposition_semitones(part_name: str) -> int:
    """
    Infer written-to-sounding transposition from Beethoven part names.

    MuseData parsing does not attach Instrument transposition metadata here, so
    we convert common transposing parts manually before writing MusicXML.
    Positive values sound higher than written; negative values sound lower.
    """
    name = (part_name or "").lower().replace("\\0", "b")

    if "flauto piccolo" in name or "piccolo" in name:
        return 12
    if "contrafagotto" in name or "contrabassoon" in name:
        return -12

    # Avoid blanket-transposing mixed cello/bass reduction parts.
    if "contrabasso" in name and "violoncello" not in name and "voiloncello" not in name:
        return -12

    if "clarinet" in name or "clarinetti" in name:
        if "in bb" in name or "in b-" in name or "in b " in name or "in b." in name:
            return -2
        if "in a" in name:
            return -3
        return 0

    if "corni" in name or "horn" in name:
        if "in eb" in name or "in e-" in name or "in e " in name or "in e." in name:
            return -9
        if "in c" in name:
            return 0

    if "trombe" in name or "trumpet" in name:
        if "in c" in name:
            return 0

    return 0


def apply_concert_transposition(part: stream.Stream) -> stream.Stream:
    """Return a part transposed from written pitch to concert pitch."""
    semitones = infer_transposition_semitones(getattr(part, "partName", "") or getattr(part, "id", ""))
    if semitones == 0:
        return part
    transposed = part.transpose(interval.Interval(semitones), inPlace=False)
    transposed.id = getattr(part, "id", None)
    transposed.partName = getattr(part, "partName", None)
    return transposed


def load_beethoven_movement(
    movement_num: int,
    data_dir: str = str(DEFAULT_DATA_DIR),
    strict: bool = False,
) -> stream.Score:
    """
    Combine all parseable Kern part files for one movement into a Score.

    Args:
        movement_num: 1-4.
        data_dir: Root folder containing movement directories.
        strict: Raise on the first failed part when True; otherwise warn/skip.
    """
    movement_path = _movement_dir(Path(data_dir), movement_num)
    score = stream.Score(id=f"Beethoven5_Movement{movement_num}")
    failures = []

    for path in _part_files(movement_path):
        try:
            parsed = load_kern_file(str(path))
            parts = list(parsed.parts) if hasattr(parsed, "parts") and parsed.parts else [parsed]
            for part in parts:
                part.id = path.name
                if not getattr(part, "partName", None):
                    part.partName = path.name
                part = apply_concert_transposition(part)
                score.insert(0, part)
        except Exception as exc:
            failures.append((path.name, str(exc)))
            if strict:
                raise

    if failures:
        print(f"Skipped {len(failures)} part(s) in movement {movement_num}:")
        for name, error in failures[:8]:
            print(f"  {name}: {error}")
    if len(score.parts) == 0:
        raise RuntimeError(f"No Kern parts parsed for movement {movement_num}")
    return score


def inspect_kern_file(path: str) -> Dict:
    """Return quick stats for one Kern file or movement directory."""
    target = Path(path)
    if target.is_dir():
        part_paths = _part_files(target)
        return {
            "path": str(target),
            "part_files": len(part_paths),
            "parts": [p.name for p in part_paths],
        }

    parsed = load_kern_file(str(target))
    return {
        "path": str(target),
        "parts": len(parsed.parts) if hasattr(parsed, "parts") else 1,
        "highest_time": float(parsed.highestTime),
        "notes": len(list(parsed.flatten().notes)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Load/inspect CCARH Beethoven Kern data.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--movement", type=int, default=1)
    parser.add_argument("--inspect", action="store_true", help="Print movement/file stats.")
    parser.add_argument("--write-xml", help="Write combined movement as MusicXML.")
    parser.add_argument("--download", action="store_true", help="Download CCARH data first.")
    args = parser.parse_args()

    if args.download:
        download_beethoven_symphonies_kern(args.data_dir)

    movement_path = _movement_dir(Path(args.data_dir), args.movement)
    if args.inspect:
        print(inspect_kern_file(str(movement_path)))

    if args.write_xml:
        score = load_beethoven_movement(args.movement, args.data_dir)
        score.write("musicxml", fp=args.write_xml)
        print(f"Wrote {args.write_xml}")


if __name__ == "__main__":
    main()
