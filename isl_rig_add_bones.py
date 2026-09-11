#!/usr/bin/env python3
"""
isl_rig_add_bones.py
Add finger and facial bones to the ISL avatar rig.

Usage:
    blender Blender_Avatar/Man_Mesh.blend --background --python isl_rig_add_bones.py
"""

import bpy
from mathutils import Vector


def get_main_armature():
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE' and 'DEF-spine' in obj.data.bones:
            return obj
    return None


def get_main_mesh():
    best = None
    best_verts = 0
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and len(obj.data.vertices) > best_verts:
            best = obj
            best_verts = len(obj.data.vertices)
    return best


def add_bone(armature, name, head, tail, parent_name):
    bpy.ops.object.select_all(action='DESELECT')
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    
    bone = armature.data.edit_bones.new(name)
    bone.head = Vector(head)
    bone.tail = Vector(tail)
    bone.use_deform = True
    
    parent = armature.data.edit_bones.get(parent_name)
    if parent:
        bone.parent = parent
    
    bpy.ops.object.mode_set(mode='OBJECT')
    return bone


def add_fingers(armature, side):
    hand_name = f"DEF-hand.{side}"
    if hand_name not in armature.data.bones:
        print(f"ERROR: {hand_name} not found!")
        return 0
    
    hand = armature.data.bones[hand_name]
    # Direction multiplier for left/right
    mx = -1 if side == 'L' else 1
    
    added = 0
    
    # [bone_suffix, direction, lengths]
    fingers = [
        ("thumb", (mx*0.015, 0.02, 0.04), [0.025, 0.02]),
        ("f_index", (mx*0.01, 0.02, 0.05), [0.025, 0.02, 0.015]),
        ("f_middle", (0.0, 0.02, 0.05), [0.03, 0.025, 0.015]),
        ("f_ring", (mx*0.01, 0.02, 0.05), [0.025, 0.02, 0.015]),
        ("f_pinky", (mx*0.02, 0.02, 0.04), [0.02, 0.015, 0.01]),
    ]
    
    for finger_name, direction, lengths in fingers:
        parent = hand_name
        dir_vec = Vector(direction)
        
        for i, length in enumerate(lengths):
            bone_name = f"DEF-{finger_name}.{i+1}.{side}"
            
            # Calculate head position
            if i == 0:
                bone_head = Vector(hand.head_local) + dir_vec * 0.3
            else:
                prev_name = f"DEF-{finger_name}.{i}.{side}"
                prev_bone = armature.data.bones.get(prev_name)
                bone_head = Vector(prev_bone.tail) if prev_bone else Vector(hand.head_local) + dir_vec * (0.3 + i * 0.02)
            
            bone_tail = bone_head + Vector((0, 0, length))
            
            bone = add_bone(armature, bone_name, bone_head, bone_tail, parent)
            if bone:
                added += 1
                print(f"  Added: {bone_name}")
            
            parent = bone_name
    
    return added


def add_face_bones(armature):
    head_name = "head"
    if head_name not in armature.data.bones:
        print("ERROR: head bone not found!")
        return 0
    
    face_defs = [
        ("DEF-jaw", Vector((0, -0.05, 0.02)), Vector((0, -0.08, -0.02))),
        ("DEF-brow.L", Vector((-0.03, -0.02, 0.04)), Vector((-0.02, -0.01, 0.05))),
        ("DEF-brow.R", Vector((0.03, -0.02, 0.04)), Vector((0.02, -0.01, 0.05))),
        ("DEF-eyelid.L", Vector((-0.025, -0.025, 0.035)), Vector((-0.025, -0.015, 0.04))),
        ("DEF-eyelid.R", Vector((0.025, -0.025, 0.035)), Vector((0.025, -0.015, 0.04))),
        ("DEF-mouth.L", Vector((-0.02, -0.06, 0.02)), Vector((-0.015, -0.065, 0.025))),
        ("DEF-mouth.R", Vector((0.02, -0.06, 0.02)), Vector((0.015, -0.065, 0.025))),
    ]
    
    added = 0
    for name, head_pos, tail_pos in face_defs:
        bone = add_bone(armature, name, head_pos, tail_pos, head_name)
        if bone:
            added += 1
            print(f"  Added: {name}")
    
    return added


def rebuild_weights(armature, mesh):
    """Rebuild vertex weights from scratch."""
    print("\nRebuilding vertex weights...")
    
    # Clear existing groups
    mesh.vertex_groups.clear()
    
    # Get bone positions
    armature_obj = bpy.data.objects.get(armature.name)
    mesh_obj = bpy.data.objects.get(mesh.name)
    
    bone_positions = {}
    for bone in armature.data.bones:
        bone_world_pos = armature_obj.matrix_world @ bone.head_local
        bone_local_pos = mesh_obj.matrix_world.inverted() @ bone_world_pos
        bone_positions[bone.name] = bone_local_pos
    
    # Assign weights based on distance to 4 closest bones
    for vert in mesh.data.vertices:
        dists = [((vert.co - pos).length, name) for name, pos in bone_positions.items()]
        dists.sort()
        
        weights = []
        for dist, bone_name in dists[:4]:
            weight = 1.0 / (dist + 0.01) if dist > 0.001 else 1.0
            weights.append((weight, bone_name))
        
        total = sum(w for w, _ in weights)
        if total == 0:
            continue
        
        for weight, bone_name in weights:
            vg = mesh.vertex_groups.get(bone_name)
            if not vg:
                vg = mesh.vertex_groups.new(name=bone_name)
            vg.add([vert.index], weight / total, 'ADD')
    
    print(f"  Done: {len(mesh.data.vertices)} vertices")


def main():
    print("=" * 60)
    print("ISL Rig: Add Bones")
    print("=" * 60)
    
    armature = get_main_armature()
    mesh = get_main_mesh()
    
    if not armature or not mesh:
        print("ERROR: Armature or mesh not found!")
        return
    
    print(f"Armature: {armature.name} ({len(armature.data.bones)} bones)")
    print(f"Mesh: {mesh.name} ({len(mesh.data.vertices)} verts)")
    
    print("\n[1/3] Adding finger bones...")
    left = add_fingers(armature, 'L')
    right = add_fingers(armature, 'R')
    print(f"  Total finger bones: {left + right}")
    
    print("\n[2/3] Adding facial bones...")
    face = add_face_bones(armature)
    print(f"  Total face bones: {face}")
    
    print("\n[3/3] Rebuilding weights...")
    rebuild_weights(armature, mesh)
    
    # Export
    blend_path = bpy.data.filepath
    glb_path = blend_path.replace('.blend', '_isl.glb') if blend_path else 'output_isl.glb'
    
    print(f"\nExporting: {glb_path}")
    bpy.ops.export_scene.gltf(
        filepath=glb_path,
        export_format='GLB',
        export_apply=True,
        export_texcoords=True,
        export_normals=True,
        export_materials='EXPORT',
        export_skins=True,
        export_morph=True,
        export_yup=True,
        export_animations=False,
    )
    
    print(f"\nDONE! Exported: {glb_path}")
    print(f"  Bones: {len(armature.data.bones)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
