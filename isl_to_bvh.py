#!/usr/bin/env python3
"""
ISL Video to BVH Converter
Downloads ISLRTC dictionary videos and converts them to BVH motion capture files
using MediaPipe Pose estimation.

Usage:
    python isl_to_bvh.py --download    # Download ISL videos
    python isl_to_bvh.py --extract     # Extract BVH from downloaded videos
    python isl_to_bvh.py --all         # Do both
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

# ISLRTC ISL Dictionary playlist
ISL_PLAYLIST_URL = "https://youtube.com/playlist?list=PLFjydPMg4Dapq9vcdmGyHs8uJhiqMgUrX"

# Paths
BASE_DIR = Path(__file__).parent
VIDEOS_DIR = BASE_DIR / "data" / "videos"
BVH_DIR = BASE_DIR / "data" / "bvh"
METADATA_FILE = BASE_DIR / "data" / "metadata.json"

# BVH skeleton hierarchy - upper body for sign language
# Root at Hips, arms for signing, head for expression
BVH_HIERARCHY = """HIERARCHY
ROOT Hips {
    OFFSET 0.00 0.00 0.00
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    JOINT Spine {
        OFFSET 0.00 10.00 0.00
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT Neck {
            OFFSET 0.00 20.00 0.00
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT Head {
                OFFSET 0.00 10.00 0.00
                CHANNELS 3 Zrotation Xrotation Yrotation
                End Site {
                    OFFSET 0.00 5.00 0.00
                }
            }
        }
        JOINT LeftShoulder {
            OFFSET 5.00 15.00 0.00
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT LeftArm {
                OFFSET 10.00 0.00 0.00
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT LeftForeArm {
                    OFFSET 10.00 0.00 0.00
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    JOINT LeftHand {
                        OFFSET 5.00 0.00 0.00
                        CHANNELS 3 Zrotation Xrotation Yrotation
                        End Site {
                            OFFSET 5.00 0.00 0.00
                        }
                    }
                }
            }
        }
        JOINT RightShoulder {
            OFFSET -5.00 15.00 0.00
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT RightArm {
                OFFSET -10.00 0.00 0.00
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT RightForeArm {
                    OFFSET -10.00 0.00 0.00
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    JOINT RightHand {
                        OFFSET -5.00 0.00 0.00
                        CHANNELS 3 Zrotation Xrotation Yrotation
                        End Site {
                            OFFSET -5.00 0.00 0.00
                        }
                    }
                }
            }
        }
    }
}
"""


def download_isl_videos():
    """Download all ISL dictionary videos from the playlist."""
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download videos (short ones only, <30s, low res for speed)
    cmd = [
        "yt-dlp",
        "--yes-playlist",
        "-f", "best[height<=480]",
        "-o", str(VIDEOS_DIR / "%(title)s.%(ext)s"),
        "--max-duration", "30",
        ISL_PLAYLIST_URL
    ]
    
    print("Downloading ISL videos from playlist...")
    subprocess.run(cmd, check=True)
    print(f"Videos saved to {VIDEOS_DIR}")


def pose_to_bvh(video_path: Path, output_path: Path):
    """Convert a video to BVH using MediaPipe Pose."""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            frame_data = extract_joint_rotations(landmarks)
            frames.append(frame_data)
    
    cap.release()
    pose.close()
    
    if not frames:
        print(f"  WARNING: No pose detected in {video_path.name}")
        return False
    
    # Write BVH file
    write_bvh(output_path, frames, fps)
    return True


def extract_joint_rotations(landmarks) -> list:
    """
    Extract joint rotations from MediaPipe pose landmarks.
    Returns rotation angles (in degrees) for each joint.
    
    Joint order in frame_data (34 values total):
    [root_pos(3), root_rot(3), spine(3), neck(3), head(3),
     left_shoulder(3), left_arm(3), left_forearm(3), left_hand(3),
     right_shoulder(3), right_arm(3), right_forearm(3), right_hand(3)]
    """
    # Get key points as numpy arrays
    left_shoulder = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z])
    right_shoulder = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
    left_elbow = np.array([landmarks[13].x, landmarks[13].y, landmarks[13].z])
    right_elbow = np.array([landmarks[14].x, landmarks[14].y, landmarks[14].z])
    left_wrist = np.array([landmarks[15].x, landmarks[15].y, landmarks[15].z])
    right_wrist = np.array([landmarks[16].x, landmarks[16].y, landmarks[16].z])
    left_hip = np.array([landmarks[23].x, landmarks[23].y, landmarks[23].z])
    right_hip = np.array([landmarks[24].x, landmarks[24].y, landmarks[24].z])
    nose = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    
    # Calculate bone vectors
    left_upper_arm = left_elbow - left_shoulder
    left_forearm = left_wrist - left_elbow
    right_upper_arm = right_elbow - right_shoulder
    right_forearm = right_wrist - right_elbow
    
    # Convert vector to Euler angles (simplified XYZ convention)
    def vec_to_euler(v):
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            return 0.0, 0.0, 0.0
        v = v / norm
        x = np.arcsin(-v[1]) * 180 / np.pi
        y = np.arctan2(v[0], v[2]) * 180 / np.pi
        z = 0.0
        return x, y, z
    
    # Root position (hip center)
    hip_center = (left_hip + right_hip) / 2
    root_x = hip_center[0] * 100
    root_y = (1.0 - hip_center[1]) * 100  # Flip Y so up is positive
    root_z = hip_center[2] * 100
    
    # Spine rotation (hip to shoulder)
    shoulder_center = (left_shoulder + right_shoulder) / 2
    spine_vec = shoulder_center - hip_center
    spine_rot = vec_to_euler(spine_vec)
    
    # Neck/head rotation
    head_vec = nose - shoulder_center
    neck_rot = vec_to_euler(head_vec)
    
    # Arm rotations
    left_shoulder_rot = vec_to_euler(left_upper_arm)
    left_elbow_rot = vec_to_euler(left_forearm)
    left_hand_rot = (0.0, 0.0, 0.0)
    
    right_shoulder_rot = vec_to_euler(right_upper_arm)
    right_elbow_rot = vec_to_euler(right_forearm)
    right_hand_rot = (0.0, 0.0, 0.0)
    
    # Assemble frame data (34 values)
    frame_values = [
        root_x, root_y, root_z, 0.0, 0.0, 0.0,  # Hips (6)
        *spine_rot,                               # Spine (3)
        *neck_rot,                                # Neck (3)
        *neck_rot,                                # Head (3)
        *left_shoulder_rot,                       # LeftShoulder (3)
        *left_shoulder_rot,                       # LeftArm (3)
        *left_elbow_rot,                          # LeftForeArm (3)
        *left_hand_rot,                           # LeftHand (3)
        *right_shoulder_rot,                      # RightShoulder (3)
        *right_shoulder_rot,                      # RightArm (3)
        *right_elbow_rot,                         # RightForeArm (3)
        *right_hand_rot,                          # RightHand (3)
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
        # Get word name from filename
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
    
    # Save metadata
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
        download_isl_videos()
    
    if args.extract or args.all:
        extract_all_bvh()
    
    if not any([args.download, args.extract, args.all, args.video]):
        parser.print_help()


if __name__ == "__main__":
    main()
