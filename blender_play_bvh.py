#!/usr/bin/env python3
"""
blender_play_bvh.py
Load the ISL avatar GLB and play BVH animation in Blender.

Usage:
    blender --python blender_play_bvh.py -- --glb Blender_Avatar/man_realistic_bust.glb --bvh data/bvh/call_it_a_day.bvh
"""

import bpy
import sys
import argparse
from pathlib import Path


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def import_glb(filepath):
    bpy.ops.import_scene.gltf(filepath=filepath, import_pack_images=True)
    armature = None
    for obj in bpy.context.selected_objects:
        if obj.type == 'ARMATURE':
            armature = obj
            break
    return armature


def import_bvh(filepath):
    bpy.ops.import_anim.bvh(filepath=filepath, global_scale=0.01, frame_start=1)
    for obj in bpy.context.selected_objects:
        if obj.type == 'ARMATURE':
            return obj
    return None


def retarget_animation(src, dst):
    if not src.animation_data or not src.animation_data.action:
        return
    dst.animation_data_create()
    dst.animation_data.action = src.animation_data.action.copy()
    dst.animation_data.action.name = "ISL_Animation"
    print(f"  Copied: {dst.animation_data.action.name}")


def main():
    argv = sys.argv
    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1:]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", default="Blender_Avatar/man_realistic_bust.glb")
    parser.add_argument("--bvh", default="data/bvh/call_it_a_day.bvh")
    args = parser.parse_args(argv)
    
    print("=" * 60)
    print("ISL Avatar Animation Player")
    print("=" * 60)
    
    # Clear scene
    clear_scene()
    
    # Import GLB
    print(f"\nImporting GLB: {args.glb}")
    glb_path = Path(args.glb)
    if not glb_path.exists():
        print(f"ERROR: {glb_path} not found!")
        return
    
    armature = import_glb(str(glb_path))
    if not armature:
        print("ERROR: No armature in GLB!")
        return
    print(f"  {armature.name}: {len(armature.data.bones)} bones")
    
    # Import BVH
    print(f"\nImporting BVH: {args.bvh}")
    bvh_path = Path(args.bvh)
    if not bvh_path.exists():
        print(f"ERROR: {bvh_path} not found!")
        return
    
    bvh_armature = import_bvh(str(bvh_path))
    if not bvh_armature:
        print("ERROR: BVH import failed!")
        return
    print(f"  {bvh_armature.name}: {bvh_armature.animation_data.action.frame_range[1]} frames")
    
    # Retarget
    print("\nRetargeting animation...")
    retarget_animation(bvh_armature, armature)
    
    # Delete BVH armature
    bpy.data.objects.remove(bvh_armature, do_unlink=True)
    
    # Setup viewport
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'
    
    print("\n" + "=" * 60)
    print("DONE! Press SPACE to play animation")
    print("=" * 60)


if __name__ == "__main__":
    main()
