"""Sample glTF animation channels in the native sprite executor."""

import math
from bisect import bisect_right
from dataclasses import dataclass
from itertools import pairwise

from .errors import NanoodleError


def animation_selection(animations, clip, frames):
    if type(frames) is not int or not 1 <= frames <= 64:
        raise NanoodleError("Animation frames must be an integer from 1 to 64.")
    if not clip and frames == 1:
        return None, None
    names = [anim.get("name") or f"Animation {i + 1}" for i, anim in enumerate(animations)]
    if clip and clip in names:
        return clip, animations[names.index(clip)]
    if not clip and names:
        return names[0], animations[0]
    available = ", ".join(names) or "(none)"
    if clip:
        raise NanoodleError(f"Unknown animation clip {clip!r}. Available clips: {available}.")
    raise NanoodleError("Multiple animation frames require a clip. Available clips: (none). Use frames=1 for a static pose.")


def unit_quat(value):
    length = math.hypot(*value)
    if not length or not math.isfinite(length):
        raise NanoodleError("Invalid glTF rotation quaternion.")
    return tuple(x / length for x in value)


def _slerp(a, b, u):
    a, b = unit_quat(a), unit_quat(b)
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0:
        b, dot = tuple(-x for x in b), -dot
    angle = math.acos(min(1.0, dot))
    left, right = 1 - u, u
    if angle > 1e-5:
        scale = math.sin(angle)
        left, right = math.sin(left * angle) / scale, math.sin(right * angle) / scale
    return unit_quat(tuple(left * x + right * y for x, y in zip(a, b)))


@dataclass(slots=True)
class Track:
    node: int
    path: str
    times: list
    values: list
    mode: str

    def sample(self, time):
        i = max(0, bisect_right(self.times, time) - 1)
        cubic = self.mode == "CUBICSPLINE"
        a = self.values[3*i + 1 if cubic else i]
        if i + 1 == len(self.times) or time <= self.times[i] or self.mode == "STEP":
            return unit_quat(a) if self.path == "rotation" else a
        dt = self.times[i + 1] - self.times[i]
        u = (time - self.times[i]) / dt
        if cubic:
            b = self.values[3*i + 4]
            out_tangent, in_tangent = self.values[3*i + 2], self.values[3*i + 3]
            u2, u3 = u * u, u * u * u
            p, q = 2*u3 - 3*u2 + 1, -2*u3 + 3*u2
            r, s = (u3 - 2*u2 + u) * dt, (u3 - u2) * dt
            v = tuple(p*x + r*m + q*y + s*n for x, m, y, n in zip(a, out_tangent, b, in_tangent))
            return unit_quat(v) if self.path == "rotation" else v
        b = self.values[i + 1]
        if self.path == "rotation":
            return _slerp(a, b, u)
        return tuple(x + u*(y - x) for x, y in zip(a, b))


def compile_tracks(scene, anim):
    tracks = []
    seen = set()
    for channel in anim["channels"]:
        target = channel["target"]
        if "node" not in target:  # glTF ignores channels without a core target.
            continue
        node, path = target["node"], target["path"]
        node_spec = scene.get("nodes", node)
        if path not in ("translation", "rotation", "scale", "weights") or (node, path) in seen:
            raise NanoodleError("Invalid or duplicate glTF animation target.")
        seen.add((node, path))
        if path != "weights" and "matrix" in node_spec:
            raise NanoodleError("A glTF animation cannot target TRS on a node with a matrix.")
        sampler_id = channel.get("sampler")
        samplers = anim["samplers"]
        if type(sampler_id) is not int or not 0 <= sampler_id < len(samplers):
            raise NanoodleError("Invalid glTF animation sampler.")
        sampler = samplers[sampler_id]
        mode = sampler.get("interpolation", "LINEAR")
        if mode not in ("STEP", "LINEAR", "CUBICSPLINE"):
            raise NanoodleError(f"Unsupported glTF animation interpolation: {mode}.")
        time_index = sampler["input"]
        if scene.get("accessors", time_index)["componentType"] != 5126:
            raise NanoodleError("glTF animation input times must use floats.")
        times = [t for (t,) in scene.accessor(time_index, ("SCALAR",))]
        if times[0] < 0 or any(b <= a for a, b in pairwise(times)):
            raise NanoodleError("glTF animation times must be nonnegative and strictly increasing.")
        cubic = mode == "CUBICSPLINE"
        if cubic and len(times) < 2:
            raise NanoodleError("CUBICSPLINE animation requires two or more keyframes.")
        rows = len(times) * (3 if cubic else 1)
        output = sampler["output"]
        if path == "weights":
            mesh = scene.get("meshes", node_spec.get("mesh"))
            primitives = mesh["primitives"]
            width = len(primitives[0].get("targets", [])) if primitives else 0
            if width == 0:
                raise NanoodleError("glTF animation weights require morph targets.")
            scalars = scene.accessor(output, ("SCALAR",), rows * width)
            values = [tuple(w for (w,) in scalars[k:k+width]) for k in range(0, len(scalars), width)]
        else:
            shape = "VEC4" if path == "rotation" else "VEC3"
            values = scene.accessor(output, (shape,), rows)
        tracks.append(Track(node, path, times, values, mode))
    if not tracks:
        raise NanoodleError("The selected animation has no supported channels.")
    return tracks
