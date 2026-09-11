#!/usr/bin/env python3
"""
Subtitle-to-BVH Matcher
Maps YouTube subtitles to ISL BVH clips.

Usage:
    python subtitle_to_bvh.py --bvh-dir data/bvh --subs subtitles.vtt
    python subtitle_to_bvh.py --bvh-dir data/bvh --subs subtitles.srt --output timeline.json
"""

import argparse
import json
import re
from pathlib import Path


def parse_vtt(vtt_path: Path) -> list:
    """Parse WebVTT subtitle file, return list of {text, start, end}."""
    entries = []
    with open(vtt_path, 'r') as f:
        content = f.read()
    
    # Split into blocks (skip WEBVTT header)
    blocks = content.strip().split('\n\n')[1:]  # skip header
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 2:
            continue
        
        # Parse timestamp line
        timestamp_line = None
        for line in lines:
            if '-->' in line:
                timestamp_line = line
                break
        
        if not timestamp_line:
            continue
        
        # Parse times
        match = re.match(r'(\d+:\d+:\d+\.\d+)\s*-->\s*(\d+:\d+:\d+\.\d+)', timestamp_line)
        if not match:
            continue
        
        start = parse_time(match.group(1))
        end = parse_time(match.group(2))
        
        # Get text (everything after timestamp)
        text_lines = []
        capture = False
        for line in lines:
            if capture:
                text_lines.append(line.strip())
            elif '-->' in line:
                capture = True
        
        text = ' '.join(text_lines).strip()
        if text:
            entries.append({
                'text': text,
                'start': start,
                'end': end,
                'duration': end - start
            })
    
    return entries


def parse_srt(srt_path: Path) -> list:
    """Parse SRT subtitle file."""
    entries = []
    with open(srt_path, 'r') as f:
        content = f.read()
    
    blocks = content.strip().split('\n\n')
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        
        # Line 1: index, Line 2: timestamp, Line 3+: text
        timestamp_line = lines[1]
        match = re.match(r'(\d+:\d+:\d+,\d+)\s*-->\s*(\d+:\d+:\d+,\d+)', timestamp_line)
        if not match:
            continue
        
        start = parse_time(match.group(1).replace(',', '.'))
        end = parse_time(match.group(2).replace(',', '.'))
        text = ' '.join(lines[2:]).strip()
        
        if text:
            entries.append({
                'text': text,
                'start': start,
                'end': end,
                'duration': end - start
            })
    
    return entries


def parse_time(t: str) -> float:
    """Parse HH:MM:SS.mmm to seconds."""
    parts = t.split(':')
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(s)


def tokenize(text: str) -> list:
    """Split subtitle text into individual words."""
    # Remove HTML tags, normalize
    text = re.sub(r'<[^>]+>', '', text)
    # Lowercase and split
    words = text.lower().split()
    # Strip punctuation
    words = [re.sub(r'[^\w]', '', w) for w in words]
    return [w for w in words if w]


def match_words_to_bvh(subtitles: list, bvh_dir: Path) -> dict:
    """
    Map each subtitle's words to available BVH clips.
    Returns a timeline of {word, start, end, bvh_file}.
    """
    # Get available BVH words
    bvh_files = list(bvh_dir.glob("*.bvh"))
    bvh_words = {f.stem.lower(): f for f in bvh_files}
    
    timeline = []
    oov_words = set()  # Out-of-vocabulary words
    
    for sub in subtitles:
        words = tokenize(sub['text'])
        # Distribute time evenly across words
        word_duration = sub['duration'] / max(len(words), 1)
        
        for i, word in enumerate(words):
            start = sub['start'] + i * word_duration
            end = start + word_duration
            
            bvh_file = bvh_words.get(word)
            
            timeline.append({
                'word': word,
                'start': round(start, 3),
                'end': round(end, 3),
                'bvh_file': str(bvh_file) if bvh_file else None,
                'fingerspell': bvh_file is None
            })
            
            if not bvh_file:
                oov_words.add(word)
    
    return {
        'timeline': timeline,
        'total_words': len(timeline),
        'matched_words': sum(1 for t in timeline if not t['fingerspell']),
        'fingerspelled_words': sum(1 for t in timeline if t['fingerspell']),
        'unique_oov': list(oov_words),
        'coverage': round(sum(1 for t in timeline if not t['fingerspell']) / max(len(timeline), 1) * 100, 1)
    }


def main():
    parser = argparse.ArgumentParser(description="Match subtitles to BVH clips")
    parser.add_argument("--bvh-dir", required=True, help="Directory containing .bvh files")
    parser.add_argument("--subs", required=True, help="Subtitle file (.vtt or .srt)")
    parser.add_argument("--output", default="timeline.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    bvh_dir = Path(args.bvh_dir)
    subs_path = Path(args.subs)
    
    if not subs_path.exists():
        print(f"Error: subtitle file not found: {subs_path}")
        return
    
    if not bvh_dir.exists():
        print(f"Error: BVH directory not found: {bvh_dir}")
        return
    
    # Parse subtitles
    if subs_path.suffix.lower() == '.vtt':
        subtitles = parse_vtt(subs_path)
    elif subs_path.suffix.lower() == '.srt':
        subtitles = parse_srt(subs_path)
    else:
        print(f"Unsupported subtitle format: {subs_path.suffix}")
        return
    
    print(f"Parsed {len(subtitles)} subtitle entries")
    
    # Match to BVH
    result = match_words_to_bvh(subtitles, bvh_dir)
    
    # Save result
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults:")
    print(f"  Total words: {result['total_words']}")
    print(f"  Matched to BVH: {result['matched_words']}")
    print(f"  Fingerspelled: {result['fingerspelled_words']}")
    print(f"  Coverage: {result['coverage']}%")
    print(f"  Unique OOV words: {len(result['unique_oov'])}")
    print(f"\nTimeline saved to: {args.output}")


if __name__ == "__main__":
    main()
