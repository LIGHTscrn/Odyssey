#!/usr/bin/env python3
"""Render ISL animation to video - all in one script."""
import bpy
import os

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import GLB
print("Importing GLB...")
bpy.ops.import_scene.gltf(filepath="/home/cachyos/Odyssey/Blender_Avatar/man_realistic_bust.glb")

# Import BVH
print("Importing BVH...")
bpy.ops.import_anim.bvh(filepath="/home/cachyos/Odyssey/data/bvh/call_it_a_day.bvh", global_scale=0.01)

# Find armatures
glb_arm = None
bvh_arm = None
for obj in bpy.context.selected_objects:
    if obj.type == 'ARMATURE':
        bone_names = [b.name for b in obj.data.bones]
        if any(n.startswith('DEF-') for n in bone_names):
            glb_arm = obj
            print(f"  Found rig: {obj.name} ({len(obj.data.bones)} bones)")
        elif obj.animation_data:
            bvh_arm = obj
            print(f"  Found BVH: {obj.name}")

# Retarget animation
if glb_arm and bvh_arm and bvh_arm.animation_data:
    glb_arm.animation_data_create()
    glb_arm.animation_data.action = bvh_arm.animation_data.action.copy()
    glb_arm.animation_data.action.name = "ISL_Animation"
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = int(bvh_arm.animation_data.action.frame_range[1])
    bpy.data.objects.remove(bvh_arm, do_unlink=True)
    print(f"Animation retargeted! {bpy.context.scene.frame_end} frames")

# Add camera
bpy.ops.object.camera_add(location=(2.5, -3, 1.5))
camera = bpy.context.active_object
camera.rotation_euler = (1.1, 0, 0.6)
bpy.context.scene.camera = camera

# Add light
bpy.ops.object.light_add(type='SUN', location=(2, -3, 5))
light = bpy.context.active_object
light.data.energy = 2.0

# Setup render
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = "/home/cachyos/Odyssey/isl_frame_"
scene.render.fps = 25

# Render
print("Rendering frames...")
bpy.ops.render.render(animation=True)
print("Done! Converting to video...")

# Convert to video
os.system("ffmpeg -y -framerate 25 -i /home/cachyos/Odyssey/isl_frame_%04d.png -c:v libx264 -pix_fmt yuv420p /home/cachyos/Odyssey/isl_animation.mp4")
print("Video saved: /home/cachyos/Odyssey/isl_animation.mp4")
