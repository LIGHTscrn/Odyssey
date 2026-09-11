#!/usr/bin/env python3
"""
ISL Video to BVH Converter (MediaPipe 33 landmarks → 222-joint GLB skeleton)
Downloads ISLRTC dictionary videos and converts them to BVH motion capture files.

Usage:
    python isl_to_bvh.py --download --playlist idioms
    python isl_to_bvh.py --extract
    python isl_to_bvh.py --all
    python isl_to_bvh.py --video path/to/video.mp4 --output output.bvh
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm
import cv2
import mediapipe as mp
import numpy as np

# ISLRTC ISL playlists
ISL_PLAYLISTS = {
    "everyday_terms": "https://youtube.com/playlist?list=PLFjydPMg4Dapq9vcdmGyHs8uJhiqMgUrX",
    "idioms": "https://youtube.com/playlist?list=PLFjydPMg4DarxA0Pe3EbAlKP2uknntTRe",
}

# Paths
BASE_DIR = Path(__file__).parent
VIDEOS_DIR = BASE_DIR / "data" / "videos"
BVH_DIR = BASE_DIR / "data" / "bvh"
METADATA_FILE = BASE_DIR / "data" / "metadata.json"

# MediaPipe Pose has 33 landmarks - map to GLB skeleton joints
# Reference: https://google.github.io/mediapipe/solutions/pose
MP_LANDMARKS = {
    "nose": 0,
    "left_eye_inner": 1, "left_eye": 2, "left_eye_outer": 3,
    "right_eye_inner": 4, "right_eye": 5, "right_eye_outer": 6,
    "left_ear": 7, "right_ear": 8,
    "mouth_left": 9, "mouth_right": 10,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_pinky": 17, "right_pinky": 18,
    "left_index": 19, "right_index": 20,
    "left_thumb": 21, "right_thumb": 22,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}

# BVH skeleton matching GLB's Mixamo rig (upper body focused for ISL)
# Hierarchy: hips → spine → chest → neck → head
#                     → shoulders → upper arms → forearms → hands
BVH_HIERARCHY = """HIERARCHY
ROOT hips {
    OFFSET 0.00 0.00 0.00
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    JOINT spine {
        OFFSET 0.00 10.00 0.00
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT spine.001 {
            OFFSET 0.00 10.00 0.00
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT spine.002 {
                OFFSET 0.00 10.00 0.00
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT spine.003 {
                    OFFSET 0.00 10.00 0.00
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    JOINT neck {
                        OFFSET 0.00 10.00 0.00
                        CHANNELS 3 Zrotation Xrotation Yrotation
                        JOINT head {
                            OFFSET 0.00 8.00 0.00
                            CHANNELS 3 Zrotation Xrotation Yrotation
                            End Site {
                                OFFSET 0.00 5.00 0.00
                            }
                        }
                    }
                    JOINT shoulder_L {
                        OFFSET 5.00 8.00 0.00
                        CHANNELS 3 Zrotation Xrotation Yrotation
                        JOINT upper_arm_L {
                            OFFSET 10.00 0.00 0.00
                            CHANNELS 3 Zrotation Xrotation Yrotation
                            JOINT forearm_L {
                                OFFSET 10.00 0.00 0.00
                                CHANNELS 3 Zrotation Xrotation Yrotation
                                JOINT hand_L {
                                    OFFSET 8.00 0.00 0.00
                                    CHANNELS 3 Zrotation Xrotation Yrotation
                                    JOINT palm_01_L {
                                        OFFSET 3.00 0.00 0.00
                                        CHANNELS 3 Zrotation Xrotation Yrotation
                                        JOINT f_index_01_L {
                                            OFFSET 4.00 0.00 0.00
                                            CHANNELS 3 Zrotation Xrotation Yrotation
                                            JOINT f_index_02_L {
                                                OFFSET 3.00 0.00 0.00
                                                CHANNELS 3 Zrotation Xrotation Yrotation
                                                JOINT f_index_03_L {
                                                    OFFSET 2.00 0.00 0.00
                                                    CHANNELS 3 Zrotation Xrotation Yrotation
                                                    End Site {
                                                        OFFSET 2.00 0.00 0.00
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    JOINT palm_02_L {
                                        OFFSET 3.00 -1.00 0.00
                                        CHANNELS 3 Zrotation Xrotation Yrotation
                                        JOINT f_middle_01_L {
                                            OFFSET 4.00 0.00 0.00
                                            CHANNELS 3 Zrotation Xrotation Yrotation
                                            JOINT f_middle_02_L {
                                                OFFSET 3.00 0.00 0.00
                                                CHANNELS 3 Zrotation Xrotation Yrotation
                                                JOINT f_middle_03_L {
                                                    OFFSET 2.00 0.00 0.00
                                                    CHANNELS 3 Zrotation Xrotation Yrotation
                                                    End Site {
                                                        OFFSET 2.00 0.00 0.00
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    JOINT palm_03_L {
                                        OFFSET 3.00 -2.00 0.00
                                        CHANNELS 3 Zrotation Xrotation Yrotation
                                        JOINT f_ring_01_L {
                                            OFFSET 4.00 0.00 0.00
                                            CHANNELS 3 Zrotation Xrotation Yrotation
                                            JOINT f_ring_02_L {
                                                OFFSET 3.00 0.00 0.00
                                                CHANNELS 3 Zrotation Xrotation Yrotation
                                                JOINT f_ring_03_L {
                                                    OFFSET 2.00 0.00 0.00
                                                    CHANNELS 3 Zrotation Xrotation Yrotation
                                                    End Site {
                                                        OFFSET 2.00 0.00 0.00
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    JOINT palm_04_L {
                                        OFFSET 3.00 -3.00 0.00
                                        CHANNELS 3 Zrotation Xrotation Yrotation
                                        JOINT f_pinky_01_L {
                                            OFFSET 3.00 0.00 0.00
                                            CHANNELS 3 Zrotation Xrotation Yrotation
                                            JOINT f_pinky_02_L {
                                                OFFSET 2.00 0.00 0.00
                                                CHANNELS 3 Zrotation Xrotation Yrotation
                                                JOINT f_pinky_03_L {
                                                    OFFSET 2.00 0.00 0.00
                                                    CHANNELS 3 Zrotation Xrotation Yrotation
                                                    End Site {
                                                        OFFSET 1.00 0.00 0.00
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    JOINT thumb_01_L {
                                        OFFSET 2.00 2.00 0.00
                                        CHANNELS 3 Zrotation Xrotation Yrotation
                                        JOINT thumb_02_L {
                                            OFFSET 3.00 0.00 0.00
                                            CHANNELS 3 Zrotation Xrotation Yrotation
                                            JOINT thumb_03_L {
                                                OFFSET 2.00 0.00 0.00
                                                CHANNELS 3 Zrotation Xrotation Yrotation
                                                End Site {
                                                    OFFSET 2.00 0.00 0.00
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    JOINT shoulder_R {
                        OFFSET -5.00 8.00 0.00
                        CHANNELS 3 Zrotation Xrotation Yrotation
                        JOINT upper_arm_R {
                            OFFSET -10.00 0.00 0.00
                            CHANNELS 3 Zrotation Xrotation Yrotation
                            JOINT forearm_R {
                                OFFSET -10.00 0.00 0.00
                                CHANNELS 3 Zrotation Xrotation Yrotation
                                JOINT hand_R {
                                    OFFSET -8.00 0.00 0.00
                                    CHANNELS 3 Zrotation Xrotation Yrotation
                                    JOINT palm_01_R {
                                        OFFSET -3.00 0.00 0.00
                                        CHANNELS 3 Zrotation Xrotation Yrotation
                                        JOINT f_index_01_R {
                                            OFFSET -4.00 0.00 0.00
                                            CHANNELS 3 Zrotation Xrotation Yrotation
                                            JOINT f_index_02_R {
                                                OFFSET -3.00 0.00 0.00
                                                CHANNELS 3 Zrotation Xrotation Yrotation
                                                JOINT f_index_03_R {
                                                    OFFSET -2.00 0.00 0.00
                                                    CHANNELS 3 Zrotation Xrotation Yrotation
                                                    End Site {
                                                        OFFSET -2.00 0.00 0.00
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    JOINT palm_02_R {
                                        OFFSET -3.00 -1.00 0.00
                                        CHANNELS 3 Zrotation Xrotation Yrotation
                                        JOINT f_middle_01_R {
                                            OFFSET -4.00 0.00 0.00
                                            CHANNELS 3 Zrotation Xrotation Yrotation
                                            JOINT f_middle_02_R {
                                                OFFSET -3.00 0.00 0.00
                                                CHANNELS 3 Zrotation Xrotation Yrotation
                                                JOINT f_middle_03_R {
                                                    OFFSET -2.00 0.00 0.00
                                                    CHANNELS 3 Zrotation Xrotation Yrotation
                                                    End Site {
                                                        OFFSET -2.00 0.00 0.00
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    JOINT palm_03_R {
                                        OFFSET -3.00 -2.00 0.00
                                        CHANNELS 3 Zrotation Xrotation Yrotation
                                        JOINT f_ring_01_R {
                                            OFFSET -4.00 0.00 0.00
                                            CHANNELS 3 Zrotation Xrotation Yrotation
                                            JOINT f_ring_02_R {
                                                OFFSET -3.00 0.00 0.00
                                                CHANNELS 3 Zrotation Xrotation Yrotation
                                                JOINT f_ring_03_R {
                                                    OFFSET -2.00 0.00 0.00
                                                    CHANNELS 3 Zrotation Xrotation Yrotation
                                                    End Site {
                                                        OFFSET -2.00 0.00 0.00
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    JOINT palm_04_R {
                                        OFFSET -3.00 -3.00 0.00
                                        CHANNELS 3 Zrotation Xrotation Yrotation
                                        JOINT f_pinky_01_R {
                                            OFFSET -3.00 0.00 0.00
                                            CHANNELS 3 Zrotation Xrotation Yrotation
                                            JOINT f_pinky_02_R {
                                                OFFSET -2.00 0.00 0.00
                                                CHANNELS 3 Zrotation Xrotation Yrotation
                                                JOINT f_pinky_03_R {
                                                    OFFSET -2.00 0.00 0.00
                                                    CHANNELS 3 Zrotation Xrotation Yrotation
                                                    End Site {
                                                        OFFSET -1.00 0.00 0.00
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    JOINT thumb_01_R {
                                        OFFSET -2.00 2.00 0.00
                                        CHANNELS 3 Zrotation Xrotation Yrotation
                                        JOINT thumb_02_R {
                                            OFFSET -3.00 0.00 0.00
                                            CHANNELS 3 Zrotation Xrotation Yrotation
                                            JOINT thumb_03_R {
                                                OFFSET -2.00 0.00 0.00
                                                CHANNELS 3 Zrotation Xrotation Yrotation
                                                End Site {
                                                    OFFSET -2.00 0.00 0.00
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
"""


def download_isl_videos(playlist_name: str = None):
    """Download ISL dictionary videos. If playlist_name given, download that one; otherwise download all."""
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    
    if playlist_name:
        if playlist_name not in ISL_PLAYLISTS:
            print(f"Unknown playlist: {playlist_name}. Available: {list(ISL_PLAYLISTS.keys())}")
            return
        urls = [(playlist_name, ISL_PLAYLISTS[playlist_name])]
    else:
        urls = list(ISL_PLAYLISTS.items())
    
    for name, url in urls:
        cmd = [
            "yt-dlp",
            "--yes-playlist",
            "-f", "134",  # 640x360 mp4 - avoids m3u8 issues
            "-o", str(VIDEOS_DIR / "%(title)s.%(ext)s"),
            url
        ]
        
        print(f"Downloading ISL playlist '{name}'...")
        subprocess.run(cmd, check=True)
    
    print(f"\nVideos saved to {VIDEOS_DIR}")


def pose_to_bvh(video_path: Path, output_path: Path):
    """Convert video to BVH using MediaPipe Pose (33 landmarks → 222-joint skeleton)."""
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    # Download model if needed
    model_path = BASE_DIR / 'pose_landmarker_full.task'
    if not model_path.exists():
        print(f"Downloading pose model...")
        import urllib.request
        url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'
        urllib.request.urlretrieve(url, str(model_path))
    
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    landmarker = vision.PoseLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    frames = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_idx * 1000 / fps)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            frame_data = extract_joint_rotations(landmarks)
            frames.append(frame_data)
        
        frame_idx += 1
    
    cap.release()
    landmarker.close()
    
    if not frames:
        print(f"  WARNING: No pose detected in {video_path.name}")
        return False
    
    write_bvh(output_path, frames, fps)
    return True


def extract_joint_rotations(landmarks) -> list:
    """
    Extract joint rotations from 33 MediaPipe pose landmarks.
    Maps to 22-joint upper body skeleton (matching GLB's Mixamo rig).
    
    Returns rotation angles (in degrees) for each joint.
    Joint order: 21 joints × 3 channels = 63 values
    [hips_pos(3), hips_rot(3), spine(3), spine.001(3), spine.002(3), spine.003(3),
     neck(3), head(3), shoulder_L(3), upper_arm_L(3), forearm_L(3), hand_L(3),
     shoulder_R(3), upper_arm_R(3), forearm_R(3), hand_R(3)]
    """
    # Convert landmarks to numpy arrays
    def lm(idx):
        return np.array([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
    
    # Get key points
    left_shoulder = lm(11)
    right_shoulder = lm(12)
    left_elbow = lm(13)
    right_elbow = lm(14)
    left_wrist = lm(15)
    right_wrist = lm(16)
    left_index = lm(19)
    right_index = lm(20)
    left_thumb = lm(21)
    right_thumb = lm(22)
    left_hip = lm(23)
    right_hip = lm(24)
    nose = lm(0)
    left_ear = lm(7)
    right_ear = lm(8)
    mouth_left = lm(9)
    mouth_right = lm(10)
    
    # Vector to Euler angles
    def vec_to_euler(v):
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            return 0.0, 0.0, 0.0
        v = v / norm
        x = np.arcsin(-v[1]) * 180 / np.pi
        y = np.arctan2(v[0], v[2]) * 180 / np.pi
        z = 0.0
        return x, y, z
    
    # Root position (hip center, flipped Y)
    hip_center = (left_hip + right_hip) / 2
    root_x = hip_center[0] * 100
    root_y = (1.0 - hip_center[1]) * 100
    root_z = hip_center[2] * 100
    
    # Spine rotation (hip to shoulder line)
    shoulder_center = (left_shoulder + right_shoulder) / 2
    spine_vec = shoulder_center - hip_center
    spine_rot = vec_to_euler(spine_vec)
    
    # Neck/head rotation
    head_vec = nose - shoulder_center
    neck_rot = vec_to_euler(head_vec)
    
    # Left arm chain
    left_upper_arm = left_elbow - left_shoulder
    left_forearm = left_wrist - left_elbow
    left_hand_dir = left_index - left_wrist
    
    left_shoulder_rot = vec_to_euler(left_upper_arm)
    left_elbow_rot = vec_to_euler(left_forearm)
    left_hand_rot = vec_to_euler(left_hand_dir)
    
    # Right arm chain
    right_upper_arm = right_elbow - right_shoulder
    right_forearm = right_wrist - right_elbow
    right_hand_dir = right_index - right_wrist
    
    right_shoulder_rot = vec_to_euler(right_upper_arm)
    right_elbow_rot = vec_to_euler(right_forearm)
    right_hand_rot = vec_to_euler(right_hand_dir)
    
    # Assemble frame data (21 joints × 3 channels = 63 values)
    frame_values = [
        root_x, root_y, root_z, 0.0, 0.0, 0.0,  # hips (6)
        *spine_rot,                               # spine (3)
        *spine_rot,                               # spine.001 (3)
        *spine_rot,                               # spine.002 (3)
        *spine_rot,                               # spine.003 (3)
        *neck_rot,                                # neck (3)
        *neck_rot,                                # head (3)
        *left_shoulder_rot,                       # shoulder_L (3)
        *left_shoulder_rot,                       # upper_arm_L (3)
        *left_elbow_rot,                          # forearm_L (3)
        *left_hand_rot,                           # hand_L (3)
        *right_shoulder_rot,                      # shoulder_R (3)
        *right_shoulder_rot,                      # upper_arm_R (3)
        *right_elbow_rot,                         # forearm_R (3)
        *right_hand_rot,                          # hand_R (3)
    ]
    
    return frame_values


def write_bvh(output_path: Path, frames: list, fps: float):
    """Write motion data to BVH file."""
    num_frames = len(frames)
    frame_time = 1.0 / fps
    
    with open(output_path, 'w') as f:
        f.write(BVH_HIERARCHY)
        f.write(f"\nMOTION\n")
        f.write(f"Frames: {num_frames}\n")
        f.write(f"Frame Time: {frame_time:.6f}\n")
        
        for frame in frames:
            line = " ".join(f"{v:.6f}" for v in frame)
            f.write(line + "\n")


def extract_all_bvh():
    """Extract BVH from all downloaded videos."""
    BVH_DIR.mkdir(parents=True, exist_ok=True)
    
    video_files = list(VIDEOS_DIR.glob("*.mp4")) + list(VIDEOS_DIR.glob("*.webm"))
    
    if not video_files:
        print("No videos found. Run with --download first.")
        return
    
    print(f"Processing {len(video_files)} videos...")
    
    success = 0
    failed = 0
    
    for video_path in tqdm(video_files, desc="Extracting BVH"):
        word = video_path.stem
        output_path = BVH_DIR / f"{word}.bvh"
        
        try:
            if pose_to_bvh(video_path, output_path):
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR processing {video_path.name}: {e}")
            failed += 1
    
    metadata = {
        "total_videos": len(video_files),
        "successful": success,
        "failed": failed,
        "bvh_dir": str(BVH_DIR),
        "words": [f.stem for f in BVH_DIR.glob("*.bvh")]
    }
    
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDone! {success} BVH files created, {failed} failed.")
    print(f"BVH files saved to: {BVH_DIR}")
    print(f"Metadata saved to: {METADATA_FILE}")


def main():
    parser = argparse.ArgumentParser(description="ISL Video to BVH Converter")
    parser.add_argument("--download", action="store_true", help="Download ISL videos")
    parser.add_argument("--extract", action="store_true", help="Extract BVH from videos")
    parser.add_argument("--all", action="store_true", help="Download and extract")
    parser.add_argument("--playlist", type=str, help=f"Playlist to download: {list(ISL_PLAYLISTS.keys())}")
    parser.add_argument("--video", type=str, help="Process a single video file")
    parser.add_argument("--output", type=str, help="Output BVH path (for single video)")
    
    args = parser.parse_args()
    
    if args.video:
        video_path = Path(args.video)
        output_path = Path(args.output) if args.output else video_path.with_suffix('.bvh')
        print(f"Processing {video_path}...")
        pose_to_bvh(video_path, output_path)
        print(f"Saved to {output_path}")
    elif args.download or args.all:
        download_isl_videos(args.playlist)
    
    if args.extract or args.all:
        extract_all_bvh()
    
    if not any([args.download, args.extract, args.all, args.video]):
        parser.print_help()


if __name__ == "__main__":
    main()
