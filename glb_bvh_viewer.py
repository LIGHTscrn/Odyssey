#!/usr/bin/env python3
"""
ISL Avatar - GLB + BVH OpenGL Viewer
Renders a GLB avatar with BVH animation using PyOpenGL.

Usage:
    python glb_bvh_viewer.py --glb Blender_Avatar/man_realistic_bust.glb --bvh data/bvh/Call\ it\ a\ day.bvh
    python glb_bvh_viewer.py --glb Blender_Avatar/man_realistic_bust.glb --bvh data/bvh/Call\ it\ a\ day.bvh --headless
"""

import sys
import json
import struct
import argparse
import numpy as np
from pathlib import Path

import pygltflib

# ─── Math Utilities ───────────────────────────────────────────────────────────

def mat4_identity():
    return np.eye(4, dtype=np.float32)

def mat4_from_translation(t):
    m = mat4_identity()
    m[:3, 3] = t
    return m

def mat4_from_rotation_euler_xyz(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    m = mat4_identity()
    m[:3, :3] = R
    return m

def mat4_compose(translation, rotation_euler):
    T = mat4_from_translation(translation)
    R = mat4_from_rotation_euler_xyz(*rotation_euler)
    return T @ R

def mat4_look_at(eye, target, up):
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    
    view = mat4_identity()
    view[0, :3] = right
    view[1, :3] = up
    view[2, :3] = -forward
    view[0, 3] = -np.dot(right, eye)
    view[1, 3] = -np.dot(up, eye)
    view[2, 3] = np.dot(forward, eye)
    return view

def mat4_perspective(fov_deg, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fov_deg) / 2)
    nf = 1 / (near - far)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) * nf
    proj[2, 3] = 2 * far * near * nf
    proj[3, 2] = -1
    return proj


# ─── BVH Parser ───────────────────────────────────────────────────────────────

def parse_bvh(path: str) -> dict:
    """Parse BVH file into joints + motion data."""
    with open(path, 'r') as f:
        content = f.read()
    
    idx = content.index('MOTION')
    hierarchy_str = content[:idx]
    motion_str = content[idx:]
    
    # Parse motion
    lines = motion_str.strip().split('\n')
    num_frames = int(lines[1].split(':')[1].strip())
    frame_time = float(lines[2].split(':')[1].strip())
    
    motion_data = []
    for line in lines[3:]:
        values = [float(v) for v in line.strip().split()]
        if values:
            motion_data.append(values)
    
    # Parse hierarchy
    joints = []
    lines = hierarchy_str.strip().split('\n')
    stack = []
    current = None
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        tokens = line.split()
        if not tokens:
            i += 1
            continue
        
        if tokens[0] in ('ROOT', 'JOINT'):
            joint = {
                'name': tokens[1],
                'offset': [0, 0, 0],
                'channels': [],
                'children': [],
                'parent': stack[-1] if stack else None
            }
            if stack:
                stack[-1]['children'].append(joint)
            else:
                joints.append(joint)
            stack.append(joint)
            current = joint
        elif tokens[0] == 'End':
            pass
        elif tokens[0] == 'OFFSET':
            current['offset'] = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
        elif tokens[0] == 'CHANNELS':
            n = int(tokens[1])
            current['channels'] = [tokens[2+j] for j in range(n)]
        elif tokens[0] == '}':
            if stack:
                stack.pop()
        i += 1
    
    return {
        'roots': joints,
        'num_frames': num_frames,
        'frame_time': frame_time,
        'motion': np.array(motion_data, dtype=np.float32),
        'fps': 1.0 / frame_time if frame_time > 0 else 30.0
    }


def compute_bvh_world_transforms(bvh: dict, frame_idx: int) -> dict:
    """Compute world transforms for all BVH joints at a given frame."""
    if frame_idx >= len(bvh['motion']):
        frame_idx = 0
    frame_data = bvh['motion'][frame_idx]
    
    transforms = {}
    channel_offset = [0]  # Use list for mutability in nested function
    
    def compute_joint(joint, parent_world):
        name = joint['name']
        offset = np.array(joint['offset'])
        
        # Parse channels
        tx, ty, tz = 0, 0, 0
        rx, ry, rz = 0, 0, 0
        
        for ch in joint['channels']:
            if ch == 'Xposition': tx = frame_data[channel_offset[0]]
            elif ch == 'Yposition': ty = frame_data[channel_offset[0]]
            elif ch == 'Zposition': tz = frame_data[channel_offset[0]]
            elif ch == 'Xrotation': rx = np.radians(frame_data[channel_offset[0]])
            elif ch == 'Yrotation': ry = np.radians(frame_data[channel_offset[0]])
            elif ch == 'Zrotation': rz = np.radians(frame_data[channel_offset[0]])
            channel_offset[0] += 1
        
        # Local transform
        T = mat4_from_translation(offset[:3])
        R = mat4_from_rotation_euler_xyz(rx, ry, rz)
        local = R @ T
        
        # For root, also add root position
        if joint['parent'] is None:
            root_T = mat4_from_translation([tx, ty, tz])
            local = root_T @ local
        
        world = parent_world @ local
        transforms[name] = {
            'world': world,
            'local': local,
            'translation': [tx, ty, tz],
            'rotation': [rx, ry, rz]
        }
        
        for child in joint['children']:
            compute_joint(child, world)
    
    for root in bvh['roots']:
        compute_joint(root, mat4_identity())
    
    return transforms


# ─── GLB Loader ───────────────────────────────────────────────────────────────

def load_glb(path: str) -> dict:
    """Load GLB, extract mesh + skeleton."""
    gltf = pygltflib.GLTF2.load(path)
    
    # Load binary buffer
    buffer = None
    with open(path, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        version = struct.unpack('<I', f.read(4))[0]
        length = struct.unpack('<I', f.read(4))[0]
        
        chunk_length = struct.unpack('<I', f.read(4))[0]
        chunk_type = struct.unpack('<I', f.read(4))[0]
        json_data = f.read(chunk_length)
        
        chunk_length = struct.unpack('<I', f.read(4))[0]
        chunk_type = struct.unpack('<I', f.read(4))[0]
        buffer = f.read(chunk_length)
    
    def get_buffer_view(index):
        bv = gltf.bufferViews[index]
        byte_offset = bv.byteOffset or 0
        return buffer[byte_offset:byte_offset + bv.byteLength]
    
    def get_accessor(index):
        acc = gltf.accessors[index]
        data = get_buffer_view(acc.bufferView)
        
        component_type = acc.componentType
        num_components = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}[acc.type]
        
        dtype = {
            5120: np.int8, 5121: np.uint8,
            5122: np.int16, 5123: np.uint16,
            5125: np.uint32, 5126: np.float32
        }[component_type]
        
        arr = np.frombuffer(data, dtype=dtype, count=acc.count * num_components)
        arr = arr.reshape(-1, num_components)
        return arr
    
    # Extract mesh
    mesh_data = None
    if gltf.meshes:
        mesh = gltf.meshes[0]
        for primitive in mesh.primitives:
            positions = get_accessor(primitive.attributes.POSITION).astype(np.float32)
            indices = get_accessor(primitive.indices).flatten().astype(np.uint32)
            normals = get_accessor(primitive.attributes.NORMAL).astype(np.float32) if primitive.attributes.NORMAL is not None else None
            joints = get_accessor(primitive.attributes.JOINTS_0).astype(np.float32) if primitive.attributes.JOINTS_0 is not None else None
            weights = get_accessor(primitive.attributes.WEIGHTS_0).astype(np.float32) if primitive.attributes.WEIGHTS_0 is not None else None
            
            # Compute bounds
            bbox_min = positions.min(axis=0)
            bbox_max = positions.max(axis=0)
            center = (bbox_min + bbox_max) / 2
            scale = np.max(bbox_max - bbox_min)
            
            mesh_data = {
                'positions': positions,
                'indices': indices,
                'normals': normals,
                'joints': joints,
                'weights': weights,
                'bbox_min': bbox_min,
                'bbox_max': bbox_max,
                'center': center,
                'scale': scale,
            }
            break
    
    # Extract skeleton
    skeleton = None
    if gltf.skins:
        skin = gltf.skins[0]
        ibm = get_accessor(skin.inverseBindMatrices).reshape(-1, 4, 4)
        
        joints_info = []
        for j in skin.joints:
            node = gltf.nodes[j]
            joints_info.append({
                'node_index': j,
                'name': node.name,
                'translation': node.translation,
                'rotation': node.rotation,
                'matrix': node.matrix,
                'children': node.children,
                'ibm': ibm[j],
            })
        
        skeleton = {
            'joints': joints_info,
            'root': skin.joints[0] if skin.joints else None,
            'num_joints': len(skin.joints),
        }
    
    return {
        'mesh': mesh_data,
        'skeleton': skeleton,
        'gltf': gltf,
    }


# ─── BVH to GLB Retargeting ──────────────────────────────────────────────────

# Mapping from BVH joint names to GLB joint names
BVH_TO_GLB_MAP = {
    'Hips': 'hips',
    'Spine': 'spine',
    'Spine1': 'spine.001',
    'Spine2': 'spine.002',
    'Neck': 'neck',
    'Head': 'head',
    'LeftShoulder': 'shoulder.L',
    'LeftArm': 'upper_arm.L',
    'LeftForeArm': 'forearm.L',
    'LeftHand': 'hand.L',
    'RightShoulder': 'shoulder.R',
    'RightArm': 'upper_arm.R',
    'RightForeArm': 'forearm.R',
    'RightHand': 'hand.R',
    'LeftUpLeg': 'thigh.L',
    'LeftLeg': 'shin.L',
    'LeftFoot': 'foot.L',
    'RightUpLeg': 'thigh.R',
    'RightLeg': 'shin.R',
    'RightFoot': 'foot.R',
}


def compute_glb_joint_world_transforms(glb: dict) -> list:
    """Compute world transforms for GLB joints in rest pose."""
    skeleton = glb['skeleton']
    if not skeleton:
        return []
    
    num_joints = skeleton['num_joints']
    world_transforms = [mat4_identity() for _ in range(num_joints)]
    joint_map = {j['node_index']: i for i, j in enumerate(skeleton['joints'])}
    
    def compute_local(joint_info):
        t = joint_info['translation'] or [0, 0, 0]
        r = joint_info['rotation'] or [0, 0, 0, 1]
        
        if joint_info['matrix'] is not None:
            return np.array(joint_info['matrix'], dtype=np.float32).reshape(4, 4).T
        
        T = mat4_from_translation(t)
        if len(r) == 4:
            # Quaternion (x, y, z, w)
            qx, qy, qz, qw = r
            R = mat4_identity()
            R[0, 0] = 1 - 2*(qy*qy + qz*qz)
            R[0, 1] = 2*(qx*qy - qz*qw)
            R[0, 2] = 2*(qx*qz + qy*qw)
            R[1, 0] = 2*(qx*qy + qz*qw)
            R[1, 1] = 1 - 2*(qx*qx + qz*qz)
            R[1, 2] = 2*(qy*qz - qx*qw)
            R[2, 0] = 2*(qx*qz - qy*qw)
            R[2, 1] = 2*(qy*qz + qx*qw)
            R[2, 2] = 1 - 2*(qx*qx + qy*qy)
        else:
            R = mat4_from_rotation_euler_xyz(r[0], r[1], r[2]) if len(r) == 3 else mat4_identity()
        
        return T @ R
    
    def traverse(node_idx, parent_world):
        local = compute_local(glb['gltf'].nodes[node_idx])
        world = parent_world @ local
        joint_idx = joint_map.get(node_idx)
        if joint_idx is not None:
            world_transforms[joint_idx] = world
        
        for child_idx in (glb['gltf'].nodes[node_idx].children or []):
            traverse(child_idx, world)
    
    if skeleton['root'] is not None:
        traverse(skeleton['root'], mat4_identity())
    
    return world_transforms


def compute_animated_glb_transforms(glb: dict, bvh: dict, frame_idx: int) -> list:
    """Compute GLB joint world transforms driven by BVH animation."""
    skeleton = glb['skeleton']
    if not skeleton:
        return []
    
    # Get BVH world transforms
    bvh_transforms = compute_bvh_world_transforms(bvh, frame_idx)
    
    # Get GLB rest pose world transforms
    glb_rest = compute_glb_joint_world_transforms(glb)
    
    num_joints = skeleton['num_joints']
    animated = [mat4_identity() for _ in range(num_joints)]
    
    # Direct mapping: BVH joint name → GLB joint name
    BVH_TO_GLB_DIRECT = {
        'hips': 'hips',
        'spine': 'spine',
        'spine.001': 'spine.001',
        'spine.002': 'spine.002',
        'spine.003': 'spine.003',
        'neck': 'neck',
        'head': 'head',
        'shoulder_L': 'shoulder.L',
        'upper_arm_L': 'upper_arm.L',
        'forearm_L': 'forearm.L',
        'hand_L': 'hand.L',
        'shoulder_R': 'shoulder.R',
        'upper_arm_R': 'upper_arm.R',
        'forearm_R': 'forearm.R',
        'hand_R': 'hand.R',
    }
    
    for i, joint in enumerate(skeleton['joints']):
        glb_name = joint['name']
        
        # Find matching BVH joint
        bvh_name = None
        for bvhn, glbn in BVH_TO_GLB_DIRECT.items():
            if glbn == glb_name:
                bvh_name = bvhn
                break
        
        if bvh_name and bvh_name in bvh_transforms:
            # Use BVH animated transform
            animated[i] = bvh_transforms[bvh_name]['world']
        else:
            # Use rest pose
            animated[i] = glb_rest[i] if i < len(glb_rest) else mat4_identity()
    
    return animated


# ─── OpenGL Renderer ──────────────────────────────────────────────────────────

VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec4 aJoints;
layout (location = 3) in vec4 aWeights;

uniform mat4 uMVP;
uniform mat4 uBoneMatrices[224];

out vec3 vNormal;
out vec3 vPos;

void main() {
    mat4 skin = mat4(0.0);
    skin += uBoneMatrices[int(aJoints.x)] * aWeights.x;
    skin += uBoneMatrices[int(aJoints.y)] * aWeights.y;
    skin += uBoneMatrices[int(aJoints.z)] * aWeights.z;
    skin += uBoneMatrices[int(aJoints.w)] * aWeights.w;
    
    vec4 pos = skin * vec4(aPos, 1.0);
    gl_Position = uMVP * pos;
    vNormal = mat3(skin) * aNormal;
    vPos = pos.xyz;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 vNormal;
in vec3 vPos;
out vec4 FragColor;

uniform vec3 uLightDir;
uniform vec3 uViewPos;

void main() {
    vec3 norm = normalize(vNormal);
    vec3 light = normalize(uLightDir);
    float diff = max(dot(norm, light), 0.0);
    vec3 viewDir = normalize(uViewPos - vPos);
    vec3 halfDir = normalize(light + viewDir);
    float spec = pow(max(dot(norm, halfDir), 0.0), 32.0);
    vec3 color = vec3(0.3, 0.3, 0.35) + diff * vec3(0.7, 0.7, 0.75) + spec * vec3(0.2);
    FragColor = vec4(color, 1.0);
}
"""


def create_shader():
    from OpenGL.GL.shaders import compileProgram, compileShader
    vs = compileShader(VERTEX_SHADER, GL_VERTEX_SHADER)
    fs = compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    return compileProgram(vs, fs)


def render(glb_path: str, bvh_path: str, headless: bool = False):
    """Main render loop."""
    import glfw
    from OpenGL.GL import (
        GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, GL_TRIANGLES,
        GL_UNSIGNED_INT, GL_FLOAT, GL_STATIC_DRAW, GL_ARRAY_BUFFER,
        GL_ELEMENT_ARRAY_BUFFER, GL_VERTEX_SHADER, GL_FRAGMENT_SHADER,
        glClear, glClearColor, glEnable, glUseProgram, glGenVertexArrays,
        glBindVertexArray, glGenBuffers, glBindBuffer, glBufferData,
        glVertexAttribPointer, glEnableVertexAttribArray, glDrawElements,
        glDeleteVertexArrays, glDeleteBuffers, glGetUniformLocation,
        glUniformMatrix4fv, glUniform3f, GL_TRUE
    )
    from OpenGL.GL.shaders import compileProgram, compileShader
    
    print(f"Loading GLB: {glb_path}")
    glb = load_glb(glb_path)
    
    print(f"Loading BVH: {bvh_path}")
    bvh = parse_bvh(bvh_path)
    
    mesh = glb['mesh']
    skeleton = glb['skeleton']
    
    if not mesh:
        print("Error: No mesh in GLB")
        return
    if not skeleton:
        print("Error: No skeleton in GLB")
        return
    
    print(f"Mesh: {len(mesh['positions'])} verts, {len(mesh['indices'])} tris")
    print(f"Skeleton: {skeleton['num_joints']} joints")
    print(f"Animation: {bvh['num_frames']} frames @ {bvh['fps']:.1f} fps")
    
    if headless:
        print("Headless mode - data loaded successfully")
        return
    
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    
    window = glfw.create_window(1280, 720, "ISL Avatar", None, None)
    if not window:
        print("Error: Window creation failed")
        glfw.terminate()
        return
    
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.1, 0.1, 0.15, 1.0)
    
    shader = create_shader()
    
    # VAO/VBO setup
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    
    pos_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, pos_vbo)
    glBufferData(GL_ARRAY_BUFFER, mesh['positions'].nbytes, mesh['positions'], GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)
    
    if mesh['normals'] is not None:
        norm_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, norm_vbo)
        glBufferData(GL_ARRAY_BUFFER, mesh['normals'].nbytes, mesh['normals'], GL_STATIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)
    
    if mesh['joints'] is not None:
        joint_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, joint_vbo)
        glBufferData(GL_ARRAY_BUFFER, mesh['joints'].nbytes, mesh['joints'], GL_STATIC_DRAW)
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(2)
    
    if mesh['weights'] is not None:
        weight_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, weight_vbo)
        glBufferData(GL_ARRAY_BUFFER, mesh['weights'].nbytes, mesh['weights'], GL_STATIC_DRAW)
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(3)
    
    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh['indices'].nbytes, mesh['indices'], GL_STATIC_DRAW)
    
    glBindVertexArray(0)
    
    mvp_loc = glGetUniformLocation(shader, "uMVP")
    bone_loc = glGetUniformLocation(shader, "uBoneMatrices")
    light_loc = glGetUniformLocation(shader, "uLightDir")
    view_loc = glGetUniformLocation(shader, "uViewPos")
    
    # Camera
    center = mesh['center']
    scale = mesh['scale']
    camera_pos = center + np.array([0, scale * 0.3, scale * 1.5], dtype=np.float32)
    camera_target = center + np.array([0, scale * 0.1, 0], dtype=np.float32)
    
    # Animation
    frame = 0
    last_time = glfw.get_time()
    
    # Precompute rest pose for bones not driven by BVH
    glb_rest = compute_glb_joint_world_transforms(glb)
    
    # Inverse bind matrices for skinning
    ibms = [j['ibm'] for j in skeleton['joints']]
    
    # Build BVH joint name → GLB joint index map
    bvh_to_glb = {}
    for bvh_name, glb_name in BVH_TO_GLB_MAP.items():
        for i, joint in enumerate(skeleton['joints']):
            if joint['name'] == glb_name:
                bvh_to_glb[bvh_name] = i
                break
    
    # Initialize bone matrices with identity
    bone_matrices = [np.eye(4, dtype=np.float32) for _ in range(min(skeleton['num_joints'], 224))]
    
    while not glfw.window_should_close(window):
        current_time = glfw.get_time()
        delta = current_time - last_time
        
        if delta >= bvh['frame_time']:
            frame = (frame + 1) % bvh['num_frames']
            last_time = current_time
            
            # Compute animated bone matrices
            animated = compute_animated_glb_transforms(glb, bvh, frame)
            
            # Convert to skinning matrices: skin = world * inv_bind
            for i in range(min(skeleton['num_joints'], 224)):
                skin_matrix = animated[i] @ ibms[i]
                bone_matrices[i] = skin_matrix.T  # Transpose for OpenGL
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(shader)
        
        width, height = glfw.get_framebuffer_size(window)
        proj = mat4_perspective(45, width / height, 0.1, 1000)
        view = mat4_look_at(camera_pos, camera_target, np.array([0, 1, 0], dtype=np.float32))
        mvp = proj @ view
        
        glUniformMatrix4fv(mvp_loc, 1, GL_TRUE, mvp)
        glUniformMatrix4fv(bone_loc, min(skeleton['num_joints'], 224), GL_TRUE, 
                          np.concatenate([m.flatten() for m in bone_matrices[:224]]))
        glUniform3f(light_loc, 0.5, 1.0, 0.5)
        glUniform3f(view_loc, camera_pos[0], camera_pos[1], camera_pos[2])
        
        glBindVertexArray(vao)
        glDrawElements(GL_TRIANGLES, len(mesh['indices']), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        
        glfw.swap_buffers(window)
        glfw.poll_events()
        
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)
    
    glfw.terminate()


def main():
    parser = argparse.ArgumentParser(description="GLB + BVH Viewer")
    parser.add_argument("--glb", required=True)
    parser.add_argument("--bvh", required=True)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    
    render(args.glb, args.bvh, args.headless)


if __name__ == "__main__":
    main()
