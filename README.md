# Odyssey - ISL Avatar Platform

Converts YouTube video subtitles to Indian Sign Language (ISL) signing avatar.

## Pipeline

```
YouTube URL → yt-dlp (subtitles) → Phrase-aware matcher → BVH clip selector → Three.js Avatar
                     ↑                                              ↓
                  ISL Dictionary                                Avatar Renderer
                  (4000+ signs + idioms)                      (BVH-driven 3D)
```

## Playlists

- **Everyday Terms** (4,087 signs): `PLFjydPMg4Dapq9vcdmGyHs8uJhiqMgUrX`
- **English Idioms in ISL** (10 phrases): `PLFjydPMg4DarxA0Pe3EbAlKP2uknntTRe`

## Phrase matching

The subtitle matcher tries longest-phrase-first (e.g. "make a long story short" → single BVH clip) before falling back to word-by-word fingerspelling.

### `isl_to_bvh.py` - BVH Extraction
Converts ISLRTC ISL dictionary videos → BVH motion files via MediaPipe Pose.

```bash
# Download ISL videos from YouTube playlist + extract BVH
python isl_to_bvh.py --all

# Step by step
python isl_to_bvh.py --download
python isl_to_bvh.py --extract

# Single video
python isl_to_bvh.py --video path/to/video.mp4 --output output.bvh
```

### `subtitle_to_bvh.py` - Subtitle Matching
Maps subtitle words to BVH clips for playback.

```bash
python subtitle_to_bvh.py --bvh-dir data/bvh --subs subtitles.vtt --output timeline.json
```

### `avatar.html` - Avatar Renderer
Three.js + BVHLoader frontend. Open in browser to preview.

## Data

- **ISL Dictionary**: [ISLRTC Everyday Terms Playlist](https://youtube.com/playlist?list=PLFjydPMg4Dapq9vcdmGyHs8uJhiqMgUrX) (4,087 signs)
- Videos: `data/videos/`
- BVH files: `data/bvh/`
- Metadata: `data/metadata.json`

## Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## BVH Skeleton Hierarchy

```
Hips → Spine → Neck → Head
              → LeftShoulder → LeftArm → LeftForeArm → LeftHand
              → RightShoulder → RightArm → RightForeArm → RightHand
```

## License

MIT
