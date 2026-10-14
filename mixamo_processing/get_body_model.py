import argparse
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bpy_utils import init_scene, decomposeObject, getVertexGroups
from utils import merge_meshes, triangulate


def main(file, dst):
    assert os.path.isfile(file), f"FBX file does not exist: {file}"
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)

    sc = init_scene()
    # Load FBX
    bpy.ops.import_scene.fbx(filepath=file)

    # Find the skeleton independently of its FBX object name.
    armatures = [ob for ob in sc.objects if ob.type == "ARMATURE"]
    assert len(armatures) == 1, f"Expected one armature, found {len(armatures)}"
    arm_ob = armatures[0]
    # Get reference of body (may be more than one object)
    obs = [ob for ob in sc.objects if ob.type == "MESH"]
    assert obs, "FBX contains no meshes"

    """ SKELETON """
    bones = arm_ob.pose.bones
    rbones = arm_ob.data.bones
    # Kinematic tree
    bnames = [b.name for b in bones]
    parents = [bnames.index(b.parent.name) if b.parent else -1 for b in bones]
    # Rest pose skeleton
    joints = np.array([rb.matrix_local.to_translation() for rb in rbones], np.float32)
    rest_pose = np.array([rb.matrix_local.to_quaternion() for rb in rbones], np.float32)
    """ TEMPLATE AND BLEND WEIGHTS """
    vertices, faces, blend_weights = [], [], []
    for ob in obs:
        me = ob.to_mesh()
        vertices += [np.float32([v.co for v in me.vertices])]
        faces += [decomposeObject(ob)[1]]
        blend_weights += [getVertexGroups(ob, bones)]
    vertices, faces = merge_meshes(vertices, faces)
    faces = triangulate(faces)
    blend_weights = np.concatenate(blend_weights, axis=0)
    """ Scale down (Mixamo scale is x100)"""
    vertices *= 0.01
    joints *= 0.01
    """ Store """
    np.savez(
        dst,
        vertices=vertices,
        faces=faces,
        blend_weights=blend_weights,
        joints=joints,
        rest_pose=rest_pose,
        parents=parents,
    )
    sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a Mixamo FBX body to NPZ.")
    parser.add_argument("--file", required=True, help="Input FBX file")
    parser.add_argument("--dst", required=True, help="Output body.npz file")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    main(file=os.path.abspath(args.file), dst=os.path.abspath(args.dst))
