#!/usr/bin/env python3
"""
blender_bvh_export.py
Blender Python script to export the rig animation as BVH.
Matches our glb_bvh_extractor.py naming convention.

Usage:
    blender Blender_Avatar/man_realistic_bust.blend --python blender_bvh_export.py

Prerequisites:
    - Rig must have animation (keyframes) baked
    - Bone names must match ISL convention (see mapping below)
"""

import bpy
from mathutils import Vector, Quaternion, Matrix
from math import degrees


# ISL bone name mapping (Blender bone name -> BVH channel order)
ISL_BONE_MAP = {
    'DEF-spine': 'DEF-spine',
    'DEF-spine.001': 'DEF-spine.001',
    'DEF-spine.002': 'DEF-spine.002',
    'DEF-spine.003': 'DEF-spine.003',
    'DEF-spine.004': 'DEF-spine.004',
    'DEF-spine.005': 'DEF-spine.005',
    'DEF-spine.006': 'DEF-spine.006',
    'DEF-shoulder.L': 'DEF-shoulder.L',
    'DEF-upper_arm.L': 'DEF-upper_arm.L',
    'DEF-forearm.L': 'DEF-forearm.L',
    'DEF-hand.L': 'DEF-hand.L',
    'DEF-shoulder.R': 'DEF-shoulder.R',
    'DEF-upper_arm.R': 'DEF-upper_arm.R',
    'DEF-forearm.R': 'DEF-forearm.R',
    'DEF-hand.R': 'DEF-hand.R',
    'DEF-thigh.L': 'DEF-thigh.L',
    'DEF-thigh.R': 'DEF-thigh.R',
}


def get_armature():
    """Get the armature object."""
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            return obj
    return None


def get_bone_world_position(armature, bone_name, frame):
    """Get world position of a bone at a specific frame."""
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    
    pose_bone = armature.pose.bones.get(bone_name)
    if not pose_bone:
        return None
    
    return armature.matrix_world @ pose_bone.head


def get_bone_world_rotation(armature, bone_name, frame):
    """Get world rotation (Euler) of a bone at a specific frame."""
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    
    pose_bone = armature.pose.bones.get(bone_name)
    if not pose_bone:
        return None
    
    # Get rotation in world space
    if pose_bone.parent:
        # Convert to world rotation
        rot = (armature.matrix_world @ pose_bone.matrix).to_euler()
    else:
        rot = (armature.matrix_world @ pose_bone.matrix).to_euler()
    
    return rot


def get_bone_local_rotation(armature, bone_name, frame):
    """Get local rotation (Euler) of a bone at a specific frame."""
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    
    pose_bone = armature.pose.bones.get(bone_name)
    if not pose_bone:
        return None
    
    # Local rotation
    if pose_bone.rotation_mode == 'QUATERNION':
        rot = pose_bone.rotation_quaternion.to_euler()
    else:
        rot = pose_bone.rotation_euler
    
    return rot


def get_bone_offset(armature, bone_name):
    """Get bone offset (head position relative to parent)."""
    bone = armature.data.bones.get(bone_name)
    if not bone:
        return Vector((0, 0, 0))
    
    if bone.parent:
        return bone.head_local - bone.parent.head_local
    else:
        return bone.head_local


def generate_bvh_hierarchy(armature, bone_names):
    """Generate BVH HIERARCHY section."""
    lines = ["HIERARCHY"]
    
    def write_bone(bone_name, depth=0, parent_name=None):
        bone = armature.data.bones.get(bone_name)
        if not bone:
            return
        
        indent = "    " * (depth + 1)
        offset = get_bone_offset(armature, bone_name)
        
        if depth == 0 or parent_name is None:
            lines.append(f"ROOT {bone_name}")
            lines.append(f"{indent}OFFSET 0.00 0.00 0.00")
            lines.append(f"{indent}CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
        else:
            lines.append(f"JOINT {bone_name}")
            lines.append(f"{indent}OFFSET {offset.x:.4f} {offset.y:.4f} {offset.z:.4f}")
            lines.append(f"{indent}CHANNELS 3 Zrotation Xrotation Yrotation")
        
        # Find children
        children = [b.name for b in bone.children if b.name in bone_names]
        for child in children:
            write_bone(child, depth + 1, bone_name)
        
        if not bone.children:
            lines.append(f"{indent}End Site")
            lines.append(f"{indent}    OFFSET {offset.x:.4f} {offset.y:.4f} {offset.z:.4f}")
    
    # Find root bones
    root_bones = [b for b in bone_names if not armature.data.bones[b].parent]
    if not root_bones:
        # Use first bone as root
        root_bones = [bone_names[0]]
    
    for root in root_bones:
        write_bone(root, 0)
    
    return '\n'.join(lines)


def generate_bvh_motion(armature, bone_names, start_frame, end_frame, fps):
    """Generate BVH MOTION section."""
    num_frames = end_frame - start_frame + 1
    frame_time = 1.0 / fps
    
    lines = ["MOTION"]
    lines.append(f"Frames: {num_frames}")
    lines.append(f"Frame Time: {frame_time:.6f}")
    
    for frame in range(start_frame, end_frame + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        
        values = []
        
        for bone_name in bone_names:
            # Get rotation
            rot = get_bone_local_rotation(armature, bone_name, frame)
            if rot is None:
                values.extend([0.0, 0.0, 0.0])
            else:
                # BVH uses Z, X, Y order
                values.extend([
                    degrees(rot.z),
                    degrees(rot.x),
                    degrees(rot.y),
                ])
        
        lines.append(' '.join(f'{v:.6f}' for v in values))
    
    return '\n'.join(lines)


def export_bvh(filepath, armature, bone_names, start_frame, end_frame, fps):
    """Export animation to BVH file."""
    hierarchy = generate_bvh_hierarchy(armature, bone_names)
    motion = generate_bvh_motion(armature, bone_names, start_frame, end_frame, fps)
    
    with open(filepath, 'w') as f:
        f.write(hierarchy)
        f.write('\n')
        f.write(motion)
    
    print(f"BVH exported: {filepath}")
    print(f"  Frames: {end_frame - start_frame + 1}")
    print(f"  Bones: {len(bone_names)}")
    print(f"  FPS: {fps}")


def bake_animation(armature, start_frame, end_frame):
    """Bake animation to keyframes (useful for NLA or constraints)."""
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    
    # Select all bones
    for bone in armature.data.bones:
        bone.select = True
    
    # Bake
    bpy.ops.nla.bake(
        frame_start=start_frame,
        frame_end=end_frame,
        only_selected=True,
        visual_keying=True,
        clear_constraints=False,
        bake_types={'POSE'}
    )
    
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"Animation baked: frames {start_frame}-{end_frame}")


def main():
    print("=" * 60)
    print("Blender BVH Export for ISL Avatar")
    print("=" * 60)
    
    armature = get_armature()
    if not armature:
        print("ERROR: No armature found!")
        return
    
    scene = bpy.context.scene
    fps = scene.render.fps
    start_frame = scene.frame_start
    end_frame = scene.frame_end
    
    print(f"Armature: {armature.name}")
    print(f"Frame range: {start_frame}-{end_frame}")
    print(f"FPS: {fps}")
    
    # Get ISL bones that exist in the rig
    bone_names = []
    for bone in armature.data.bones:
        if bone.name in ISL_BONE_MAP:
            bone_names.append(bone.name)
    
    print(f"ISL bones found: {len(bone_names)}/{len(ISL_BONE_MAP)}")
    missing = set(ISL_BONE_MAP.keys()) - set(bone_names)
    if missing:
        print(f"Missing bones: {missing}")
    
    # Sort bones in hierarchy order
    sorted_bones = []
    def sort_hierarchy(bone_name):
        if bone_name in bone_names and bone_name not in sorted_bones:
            sorted_bones.append(bone_name)
            bone = armature.data.bones[bone_name]
            for child in bone.children:
                sort_hierarchy(child.name)
    
    # Start from roots
    for bone in armature.data.bones:
        if bone.parent is None and bone.name in bone_names:
            sort_hierarchy(bone.name)
    
    bone_names = sorted_bones
    print(f"Exporting {len(bone_names)} bones in hierarchy order")
    
    # Optional: bake animation
    # bake_animation(armature, start_frame, end_frame)
    
    # Export
    blend_path = bpy.data.filepath
    if blend_path:
        bvh_path = blend_path.replace('.blend', f'_isl_{start_frame}-{end_frame}.bvh')
    else:
        bvh_path = 'animation.bvh'
    
    export_bvh(bvh_path, armature, bone_names, start_frame, end_frame, fps)
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
