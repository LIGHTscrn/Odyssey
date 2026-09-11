#!/usr/bin/env python3
"""
glb_bvh_extractor.py — Full 222-joint BVH from video.
Drives the man_realistic_bust.glb rig with MediaPipe Pose.
Unmapped bones stay at rest pose.

Usage:
    python glb_bvh_extractor.py --video in.mp4 --output out.bvh
    python glb_bvh_extractor.py --dump-skeleton
"""

import json
import struct
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

GLB_PATH = Path(__file__).parent / "Blender_Avatar" / "man_realistic_bust.glb"
MODEL_PATH = Path(__file__).parent / "pose_landmarker_full.task"


def load_glb(glb_path: Path):
    with open(glb_path, 'rb') as f:
        magic, version, length = struct.unpack('<III', f.read(12))
        json_len, json_type = struct.unpack('<II', f.read(8))
        json_data = json.loads(f.read(json_len).decode('utf-8'))
        bin_len, bin_type = struct.unpack('<II', f.read(8))
        bin_data = f.read(bin_len)
    return json_data, bin_data


def build_skeleton(json_data, bin_data):
    skin = json_data['skins'][0]
    joints = skin['joints']
    nodes = json_data['nodes']
    accessors = json_data['accessors']
    buffer_views = json_data['bufferViews']

    ibm_accessor = accessors[skin['inverseBindMatrices']]
    bv = buffer_views[ibm_accessor['bufferView']]
    offset = bv.get('byteOffset', 0)
    count = ibm_accessor['count']

    ibms = []
    for i in range(count):
        mat_offset = offset + i * 64
        mat = struct.unpack('<16f', bin_data[mat_offset:mat_offset+64])
        ibms.append(np.array(mat).reshape(4, 4))

    joint_info = {}
    for i, joint_idx in enumerate(joints):
        node = nodes[joint_idx]
        name = node.get('name', f'joint_{joint_idx}')
        joint_info[joint_idx] = {
            'name': name,
            'index': joint_idx,
            'skin_index': i,
            'translation': np.array(node.get('translation', [0.0, 0.0, 0.0])),
            'rotation': np.array(node.get('rotation', [0.0, 0.0, 0.0, 1.0])),
            'ibm': ibms[i],
            'children': [c for c in node.get('children', []) if c in joints],
        }

    all_children = set()
    for j in joint_info.values():
        all_children.update(j['children'])
    roots = [idx for idx in joints if idx not in all_children]

    return {'joints': joint_info, 'roots': roots}


# Map MediaPipe landmarks to rig bones: {bone_name: (parent_lm, child_lm)}
# We drive upper body from MediaPipe; everything else stays at rest pose.
MEDIAPIPE_MAP = {
    'DEF-shoulder.L': (11, 13),
    'DEF-upper_arm.L': (11, 13),
    'DEF-forearm.L': (13, 15),
    'DEF-hand.L': (15, 15),
    'DEF-shoulder.R': (12, 14),
    'DEF-upper_arm.R': (12, 14),
    'DEF-forearm.R': (14, 16),
    'DEF-hand.R': (16, 16),
    'neck': (11, 0),
    'head': (0, 0),
    'chest': (11, 12),
    'hips': (23, 24),
    'DEF-spine': (23, 11),
    'DEF-thigh.L': (23, 27),
    'DEF-thigh.R': (24, 28),
}


def vec_to_euler_deg(v):
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return (0.0, 0.0, 0.0)
    v = v / norm
    return (np.arcsin(-v[1]) * 180 / np.pi,
            np.arctan2(v[0], v[2]) * 180 / np.pi,
            0.0)


def extract_bone_rotations(landmarks, skeleton):
    rotations = {}
    for bone_name, (p_lm, c_lm) in MEDIAPIPE_MAP.items():
        p1 = np.array([landmarks[p_lm].x, landmarks[p_lm].y, landmarks[p_lm].z])
        p2 = np.array([landmarks[c_lm].x, landmarks[c_lm].y, landmarks[c_lm].z])
        rotations[bone_name] = vec_to_euler_deg(p2 - p1)
    return rotations


def generate_bvh(skeleton, output_path, fps=30.0):
    """BVH with ALL 222 joints. Standard single-root format."""
    joints = skeleton['joints']
    roots = skeleton['roots']

    # Use first root as the single BVH root
    if not roots:
        return
    
    # Find the "root" bone (should be the skeleton root)
    root_idx = None
    for r in roots:
        if joints[r]['name'] == 'root':
            root_idx = r
            break
    if root_idx is None:
        root_idx = roots[0]

    bvh_lines = ["HIERARCHY"]
    
    def write_joint(idx, depth=0):
        lines = []
        j = joints[idx]
        name = j['name']
        t = j['translation']

        if depth == 0:
            lines.append(f"ROOT {name}")
            lines.append("{")
            lines.append(f"OFFSET 0.00 0.00 0.00")
            lines.append(f"CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
        else:
            lines.append(f"JOINT {name}")
            lines.append("{")
            lines.append(f"OFFSET {t[0]:.4f} {t[1]:.4f} {t[2]:.4f}")
            lines.append(f"CHANNELS 3 Zrotation Xrotation Yrotation")

        for child_idx in j['children']:
            lines.extend(write_joint(child_idx, depth + 1))

        if not j['children']:
            lines.append(f"End Site")
            lines.append("{")
            lines.append(f"OFFSET {t[0]:.4f} {t[1]:.4f} {t[2]:.4f}")
            lines.append("}")
        
        lines.append("}")

        return lines

    bvh_lines.extend(write_joint(root_idx, 0))

    with open(output_path, 'w') as f:
        f.write('\n'.join(bvh_lines))
        f.write(f"\nMOTION\nFrames: 0\nFrame Time: {1.0/fps:.6f}\n")

    print(f"BVH skeleton saved: {output_path}  ({len(joints)} joints)")


def write_joint_joint(idx, depth, joints):
    """Helper to write joint hierarchy."""
    lines = []
    j = joints[idx]
    name = j['name']
    t = j['translation']

    if depth == 0:
        lines.append(f"ROOT {name}")
        lines.append(f"OFFSET 0.00 0.00 0.00")
        lines.append(f"CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
    else:
        lines.append(f"JOINT {name}")
        lines.append(f"OFFSET {t[0]:.4f} {t[1]:.4f} {t[2]:.4f}")
        lines.append(f"CHANNELS 3 Zrotation Xrotation Yrotation")

    for child_idx in j['children']:
        lines.extend(write_joint_joint(child_idx, depth + 1, joints))

    if not j['children']:
        lines.append(f"End Site")
        lines.append(f"OFFSET {t[0]:.4f} {t[1]:.4f} {t[2]:.4f}")

    return lines


def extract_bvh_from_video(video_path, output_path, skeleton):
    if not MODEL_PATH.exists():
        print("Downloading MediaPipe model...")
        import urllib.request
        url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'
        urllib.request.urlretrieve(url, str(MODEL_PATH))

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5)
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
            frames.append(extract_bone_rotations(result.pose_landmarks[0], skeleton))
        frame_idx += 1

    cap.release()
    landmarker.close()

    if not frames:
        print("No pose detected!")
        return False

    # Get bone order from full skeleton DFS
    bone_names = []
    def collect(idx):
        bone_names.append(skeleton['joints'][idx]['name'])
        for c in skeleton['joints'][idx]['children']:
            collect(c)
    for root in skeleton['roots']:
        collect(root)

    # Build motion lines
    motion_lines = []
    for frame_rotations in frames:
        values = []
        for bone_name in bone_names:
            rot = frame_rotations.get(bone_name, (0.0, 0.0, 0.0))
            values.extend(rot)
        motion_lines.append(' '.join(f'{v:.6f}' for v in values))

    # Reconstruct file
    generate_bvh(skeleton, output_path, fps)
    with open(output_path, 'r') as f:
        lines = f.read().split('\n')
    motion_start = lines.index('MOTION')

    output_lines = lines[:motion_start + 1]
    output_lines.append(f'Frames: {len(frames)}')
    output_lines.append(f'Frame Time: {1.0/fps:.6f}')
    output_lines.extend(motion_lines)

    with open(output_path, 'w') as f:
        f.write('\n'.join(output_lines))

    print(f"BVH saved: {output_path}")
    print(f"  Frames: {len(frames)}, Joints: {len(bone_names)}, FPS: {fps:.1f}")
    return True


def main():
    parser = argparse.ArgumentParser(description="GLB → BVH (222 joints)")
    parser.add_argument("--dump-skeleton", action="store_true")
    parser.add_argument("--video", type=str)
    parser.add_argument("--output", type=str, default="output.bvh")
    parser.add_argument("--glb", type=str, default=str(GLB_PATH))

    args = parser.parse_args()
    json_data, bin_data = load_glb(Path(args.glb))
    skeleton = build_skeleton(json_data, bin_data)
    print(f"GLB skeleton: {len(skeleton['joints'])} joints")

    if args.dump_skeleton:
        for idx, j in skeleton['joints'].items():
            print(f"  {idx}: {j['name']} children={j['children']}")

    if args.video:
        extract_bvh_from_video(Path(args.video), Path(args.output), skeleton)


if __name__ == "__main__":
    main()
