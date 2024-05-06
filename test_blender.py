import bpy
import bmesh
from mathutils import Vector

# Replace 'positions' with your list of tuples
positions = [(0, 0, 0), (1, 2, 3), (4, 5, 6), (7, 8, 9)]

# Create a new mesh and a new object
mesh = bpy.data.meshes.new("CurveMesh")
obj = bpy.data.objects.new("CurveObject", mesh)

# Link the object to the scene
bpy.context.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj
obj.select_set(True)

# Use bmesh to create vertices and edges
bm = bmesh.new()
for pos in positions:
    bm.verts.new(pos)
bm.verts.ensure_lookup_table()
for i in range(len(positions) - 1):
    bm.edges.new((bm.verts[i], bm.verts[i + 1]))

# Finish up and write the bmesh back to the mesh
bm.to_mesh(mesh)
bm.free()

# Add a curve modifier to smooth the curve
bpy.ops.object.modifier_add(type='CURVE')
obj.modifiers['Curve'].object = obj  # This refers to the object itself as the curve shape
