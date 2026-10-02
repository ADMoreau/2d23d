"""A minimal glTF 2.0 binary (GLB) writer: meshes, textures, a node tree, skins and animations. Just what export needs."""

import json
import struct
from pathlib import Path

import numpy as np

COMPONENT_TYPES = {np.dtype(np.float32): 5126, np.dtype(np.uint32): 5125, np.dtype(np.uint16): 5123, np.dtype(np.uint8): 5121}
ACCESSOR_TYPES = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4", 16: "MAT4"}
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
LINEAR, LINEAR_MIPMAP_LINEAR, CLAMP_TO_EDGE = 9729, 9987, 33071


class GLB:
    def __init__(self) -> None:
        self.gltf: dict = {
            "asset": {"version": "2.0", "generator": "2d23d"},
            "scene": 0,
            "scenes": [{"nodes": []}],
            "nodes": [],
            "meshes": [],
            "materials": [],
            "textures": [],
            "images": [],
            "samplers": [],
            "skins": [],
            "cameras": [],
            "animations": [],
            "accessors": [],
            "bufferViews": [],
            "extensionsUsed": [],
        }
        self.data = bytearray()
        self.shared: dict[bytes, int] = {}  # accessors of repeated arrays, such as animation times

    def add_accessor(self, array: np.ndarray, target: int | None = None) -> int:
        """Store an array of shape (n,), (n, k) or (n, 4, 4) (matrices); return its accessor index."""
        array = np.ascontiguousarray(array)
        if array.ndim == 3:  # glTF matrices are column-major
            array = np.ascontiguousarray(array.transpose(0, 2, 1)).reshape(len(array), 16)
        self.data += bytes(-len(self.data) % 4)  # keep every view 4-byte aligned
        view = {"buffer": 0, "byteOffset": len(self.data), "byteLength": array.nbytes}
        if target is not None:
            view["target"] = target
        self.data += array.tobytes()
        self.gltf["bufferViews"].append(view)
        accessor = {
            "bufferView": len(self.gltf["bufferViews"]) - 1,
            "componentType": COMPONENT_TYPES[array.dtype],
            "count": len(array),
            "type": ACCESSOR_TYPES[1 if array.ndim == 1 else array.shape[1]],
        }
        if array.dtype == np.float32:  # required for positions and animation times, harmless elsewhere
            accessor["min"] = np.atleast_1d(array.min(axis=0)).tolist()
            accessor["max"] = np.atleast_1d(array.max(axis=0)).tolist()
        self.gltf["accessors"].append(accessor)
        return len(self.gltf["accessors"]) - 1

    def add_shared_accessor(self, array: np.ndarray) -> int:
        """Like add_accessor, but stores each distinct array once."""
        key = array.tobytes() + str(array.shape).encode()
        if key not in self.shared:
            self.shared[key] = self.add_accessor(array)
        return self.shared[key]

    def add_texture(self, image: bytes, mime_type: str = "image/jpeg") -> int:
        """A texture from an encoded image, smoothly filtered and not repeated; return its index."""
        self.data += bytes(-len(self.data) % 4)
        self.gltf["bufferViews"].append({"buffer": 0, "byteOffset": len(self.data), "byteLength": len(image)})
        self.data += image
        self.gltf["images"].append({"bufferView": len(self.gltf["bufferViews"]) - 1, "mimeType": mime_type})
        if not self.gltf["samplers"]:
            self.gltf["samplers"].append({"magFilter": LINEAR, "minFilter": LINEAR_MIPMAP_LINEAR, "wrapS": CLAMP_TO_EDGE, "wrapT": CLAMP_TO_EDGE})
        self.gltf["textures"].append({"sampler": 0, "source": len(self.gltf["images"]) - 1})
        return len(self.gltf["textures"]) - 1

    def add_material(self, color: tuple[float, ...], texture: int | None = None, unlit: bool = False, double_sided: bool = True) -> int:
        """A matte material in one RGBA color, times a texture if given; return its index.

        Unlit materials show their color as is, unshaded: for surfaces painted from a video, whose light is in the picture.
        """
        pbr = {"baseColorFactor": list(color), "metallicFactor": 0.0, "roughnessFactor": 1.0}
        if texture is not None:
            pbr["baseColorTexture"] = {"index": texture}
        material = {"pbrMetallicRoughness": pbr, "doubleSided": double_sided}
        if unlit:
            material["extensions"] = {"KHR_materials_unlit": {}}
            if "KHR_materials_unlit" not in self.gltf["extensionsUsed"]:
                self.gltf["extensionsUsed"].append("KHR_materials_unlit")
        self.gltf["materials"].append(material)
        return len(self.gltf["materials"]) - 1

    def add_mesh(
        self,
        name: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        material: int,
        normals: np.ndarray | None = None,
        joints: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        uvs: np.ndarray | None = None,
        colors: np.ndarray | None = None,
    ) -> int:
        """A triangle mesh, optionally with normals, 4 skin influences, texture coordinates and linear RGB colors per vertex; return its index."""
        attributes = {"POSITION": self.add_accessor(vertices.astype(np.float32), ARRAY_BUFFER)}
        if normals is not None:
            attributes["NORMAL"] = self.add_accessor(normals.astype(np.float32), ARRAY_BUFFER)
        if uvs is not None:
            attributes["TEXCOORD_0"] = self.add_accessor(uvs.astype(np.float32), ARRAY_BUFFER)
        if colors is not None:
            attributes["COLOR_0"] = self.add_accessor(colors.astype(np.float32), ARRAY_BUFFER)
        if joints is not None:
            attributes["JOINTS_0"] = self.add_accessor(joints.astype(np.uint8 if joints.max() < 256 else np.uint16), ARRAY_BUFFER)
            attributes["WEIGHTS_0"] = self.add_accessor(weights.astype(np.float32), ARRAY_BUFFER)
        primitive = {
            "attributes": attributes,
            "indices": self.add_accessor(faces.astype(np.uint16 if len(vertices) <= 65535 else np.uint32).ravel(), ELEMENT_ARRAY_BUFFER),
            "material": material,
        }
        self.gltf["meshes"].append({"name": name, "primitives": [primitive]})
        return len(self.gltf["meshes"]) - 1

    def add_node(self, name: str, parent: int | None = None, **properties) -> int:
        """A node (with any of mesh, skin, camera, translation, rotation, scale, extras), under `parent` or at the scene's root."""
        node = {"name": name} | {key: value for key, value in properties.items() if value is not None}
        for key in ("translation", "rotation", "scale"):
            if key in node:
                node[key] = [float(v) for v in node[key]]
        self.gltf["nodes"].append(node)
        index = len(self.gltf["nodes"]) - 1
        if parent is None:
            self.gltf["scenes"][0]["nodes"].append(index)
        else:
            self.gltf["nodes"][parent].setdefault("children", []).append(index)
        return index

    def add_camera(self, aspect: float, yfov: float, znear: float = 0.05) -> int:
        """A perspective camera (vertical field of view in radians), for a node to hold; return its index."""
        self.gltf["cameras"].append({"type": "perspective", "perspective": {"aspectRatio": aspect, "yfov": yfov, "znear": znear}})
        return len(self.gltf["cameras"]) - 1

    def add_skin(self, joints: list[int], inverse_bind_matrices: np.ndarray, skeleton: int) -> int:
        self.gltf["skins"].append(
            {"joints": joints, "inverseBindMatrices": self.add_accessor(inverse_bind_matrices.astype(np.float32)), "skeleton": skeleton}
        )
        return len(self.gltf["skins"]) - 1

    def add_animation(self, name: str, channels: list[tuple[int, str, np.ndarray, np.ndarray, str]]) -> int:
        """channels: (node, path, times, values, interpolation), path being translation, rotation or scale."""
        animation = {"name": name, "samplers": [], "channels": []}
        for node, path, times, values, interpolation in channels:
            sampler = {
                "input": self.add_shared_accessor(times.astype(np.float32)),
                "output": self.add_accessor(values.astype(np.float32)),
                "interpolation": interpolation,
            }
            animation["samplers"].append(sampler)
            animation["channels"].append({"sampler": len(animation["samplers"]) - 1, "target": {"node": node, "path": path}})
        self.gltf["animations"].append(animation)
        return len(self.gltf["animations"]) - 1

    def save(self, path: Path) -> None:
        gltf = {key: value for key, value in self.gltf.items() if value != []}  # glTF forbids empty arrays
        gltf["buffers"] = [{"byteLength": len(self.data)}]
        json_chunk = json.dumps(gltf).encode()
        json_chunk += b" " * (-len(json_chunk) % 4)
        bin_chunk = bytes(self.data) + bytes(-len(self.data) % 4)
        body = struct.pack("<I4s", len(json_chunk), b"JSON") + json_chunk
        body += struct.pack("<I4s", len(bin_chunk), b"BIN\0") + bin_chunk
        path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)
