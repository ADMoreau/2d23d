import json
import struct

import cv2
import numpy as np

from twod23d.glb import GLB
from twod23d.stages.export import floor_mesh


def read_glb(path):
    data = path.read_bytes()
    assert struct.unpack_from("<4sII", data) == (b"glTF", 2, len(data))
    json_length, chunk_type = struct.unpack_from("<I4s", data, 12)
    assert chunk_type == b"JSON"
    gltf = json.loads(data[20 : 20 + json_length])
    bin_length, chunk_type = struct.unpack_from("<I4s", data, 20 + json_length)
    assert chunk_type == b"BIN\0" and bin_length >= gltf["buffers"][0]["byteLength"]
    return gltf, data[28 + json_length :]


def test_floor_glb(tmp_path):
    glb = GLB()
    floor = glb.add_mesh("floor", *floor_mesh(np.array([-5, -5]), np.array([5, 5])), glb.add_material((0.8, 0.8, 0.8, 1)))
    glb.add_node("floor", mesh=floor)
    glb.save(tmp_path / "scene.glb")

    gltf, _ = read_glb(tmp_path / "scene.glb")
    assert [node["name"] for node in gltf["nodes"]] == ["floor"]
    position = gltf["accessors"][gltf["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    assert (position["count"], position["min"], position["max"]) == (4, [-5, 0, -5], [5, 0, 5])
    assert "skins" not in gltf and "animations" not in gltf  # empty lists are left out


def test_skinned_animated_glb(tmp_path):
    glb = GLB()
    root = glb.add_node("root")
    joints = [glb.add_node("hip", parent=root), None]
    joints[1] = glb.add_node("knee", parent=joints[0], translation=[0, -0.5, 0])
    bind = np.tile(np.eye(4), (2, 1, 1))
    bind[1, :3, 3] = [1, 2, 3]
    skin = glb.add_skin(joints, bind, skeleton=joints[0])
    vertices = np.array([[0, 0, 0], [0, -1, 0], [0.1, -1, 0]], np.float32)
    mesh = glb.add_mesh("leg", vertices, np.array([[0, 1, 2]]), glb.add_material((1, 0, 0, 1)), joints=np.array([[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]), weights=np.eye(4)[[0, 0, 0]])
    glb.add_node("leg", mesh=mesh, skin=skin)
    times = np.array([0.0, 1.0])
    glb.add_animation("motion", [(joints[1], "rotation", times, np.array([[0, 0, 0, 1], [1, 0, 0, 0]]), "LINEAR"), (root, "scale", times, np.ones((2, 3)), "STEP")])
    glb.save(tmp_path / "scene.glb")

    gltf, binary = read_glb(tmp_path / "scene.glb")
    assert gltf["nodes"][root]["children"] == [joints[0]] and gltf["nodes"][joints[0]]["children"] == [joints[1]]
    assert gltf["scenes"][0]["nodes"] == [root, 3]  # the skinned mesh's node sits at the root
    matrices = gltf["accessors"][gltf["skins"][0]["inverseBindMatrices"]]
    assert matrices["type"] == "MAT4" and matrices["count"] == 2
    view = gltf["bufferViews"][matrices["bufferView"]]
    second = np.frombuffer(binary[view["byteOffset"] + 64 : view["byteOffset"] + 128], np.float32)
    assert second[12:15].tolist() == [1, 2, 3]  # column-major: the translation is the last column
    joints_accessor = gltf["accessors"][gltf["meshes"][0]["primitives"][0]["attributes"]["JOINTS_0"]]
    assert joints_accessor["componentType"] == 5121  # unsigned bytes: few joints
    samplers = gltf["animations"][0]["samplers"]
    assert samplers[0]["input"] == samplers[1]["input"]  # the same times are stored once


def test_textured_unlit_glb(tmp_path):
    glb = GLB()
    jpeg = cv2.imencode(".jpg", np.full((4, 8, 3), 200, np.uint8))[1].tobytes()
    material = glb.add_material((1, 1, 1, 1), texture=glb.add_texture(jpeg), unlit=True, double_sided=False)
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    uvs = np.array([[0, 1], [1, 1], [0, 0]], np.float32)
    mesh = glb.add_mesh("scene", vertices, np.array([[0, 1, 2]]), material, uvs=uvs)
    glb.add_node("scene", mesh=mesh, extras={"play_area": [-1, -2, 1, 2]})
    glb.save(tmp_path / "scene.glb")

    gltf, binary = read_glb(tmp_path / "scene.glb")
    assert gltf["extensionsUsed"] == ["KHR_materials_unlit"]
    (material,) = gltf["materials"]
    assert material["extensions"] == {"KHR_materials_unlit": {}} and material["doubleSided"] is False
    assert material["pbrMetallicRoughness"]["baseColorTexture"] == {"index": 0}
    image = gltf["images"][gltf["textures"][0]["source"]]
    view = gltf["bufferViews"][image["bufferView"]]
    assert image["mimeType"] == "image/jpeg" and binary[view["byteOffset"] : view["byteOffset"] + view["byteLength"]] == jpeg
    primitive = gltf["meshes"][0]["primitives"][0]
    assert gltf["accessors"][primitive["attributes"]["TEXCOORD_0"]]["type"] == "VEC2"
    assert gltf["accessors"][primitive["indices"]]["componentType"] == 5123  # few vertices: 16-bit indices
    assert gltf["nodes"][0]["extras"] == {"play_area": [-1, -2, 1, 2]}


def test_vertex_colors(tmp_path):
    glb = GLB()
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64)
    glb.add_node("triangle", mesh=glb.add_mesh("triangle", vertices, np.array([[0, 1, 2]]), glb.add_material((1, 1, 1, 1)), colors=colors))
    glb.save(tmp_path / "scene.glb")

    gltf, binary = read_glb(tmp_path / "scene.glb")
    accessor = gltf["accessors"][gltf["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"]]
    assert (accessor["type"], accessor["componentType"], accessor["count"]) == ("VEC3", 5126, 3)
    view = gltf["bufferViews"][accessor["bufferView"]]
    assert np.frombuffer(binary[view["byteOffset"] : view["byteOffset"] + view["byteLength"]], np.float32).tolist() == colors.ravel().tolist()
