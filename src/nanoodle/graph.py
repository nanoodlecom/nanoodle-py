"""Graph loading, materialization (aliases / migrations) and topological order.

Mirrors the loader semantics of the nanoodle editor (applyGraphData) and the
exported-app runtime (play.html RUNTIME_JS runGraph step 1).
"""

import json
import re

from .errors import NanoodleError

# Dynamic input-port families (SPEC-engine execution step 4).
IMG_PORT_RE = re.compile(r"^img\d*$")        # llm vision slots, decide candidates
EDIT_IMG_RE = re.compile(r"^image\d*$")      # edit multi-reference: image, image2, ...
VID_PORT_RE = re.compile(r"^vid\d*$")        # combine clips
CLIP_PORT_RE = re.compile(r"^clip\d*$")      # combine clips (spec alias)
REF_PORT_RE = re.compile(r"^ref\d*$")        # tvideo reference images
MAX_FRAMES = 48
_FRAME_PORT_INDEX = {"frame%d" % i: i for i in range(1, MAX_FRAMES + 1)}


def wired_frames_floor(graph, node_id):
    """Highest supported frameN port wired OUT of a vframes node, at least 1.

    fields.frames is shape-affecting: run() emits frame1..frameN and downstream
    links read fixed frameK ports. A count below the highest wired port starves
    consumers mid-run (after upstream paid steps). Mirrors play.html
    wiredFramesFloor — floor is raised at run and in derive_settings.
    """
    floor = 1
    outbound = graph.outbound(node_id) if graph is not None else ()
    for link in outbound:
        if isinstance(link.from_port, str):
            floor = max(floor, _FRAME_PORT_INDEX.get(link.from_port, 1))
    return floor


# Node type registry. outputs = [(port, kind)] with the PRIMARY port first.
# extra_source_ports can be wired but don't change how terminal outputs are derived.
# static = declared input port names; dynamic = port-name regexes.
# network nodes need an API key; local media ops need ffmpeg on PATH (soft dependency).
NODE_TYPES = {
    "text":       {"title": "Text",             "outputs": [("text", "text")]},
    "upload":     {"title": "Image input",      "outputs": [("image", "image")]},
    "aupload":    {"title": "Audio input",      "outputs": [("audio", "audio")]},
    "vupload":    {"title": "Video input",      "outputs": [("video", "video")]},
    "mupload":    {"title": "3D input",         "outputs": [("model", "model3d")]},
    "choice":     {"title": "Choice",           "outputs": [("text", "text")]},
    "join":       {"title": "Join",             "outputs": [("text", "text")], "static": ["a", "b"]},
    "llm":        {"title": "LLM",              "outputs": [("text", "text")], "static": ["audio"],
                   "dynamic": [IMG_PORT_RE], "network": True},
    "image":      {"title": "Image",            "outputs": [("image", "image")],
                   "extra_source_ports": ("images",), "network": True},
    "edit":       {"title": "Edit",             "outputs": [("image", "image")],
                   "extra_source_ports": ("images", "layers"),
                   "dynamic": [EDIT_IMG_RE], "network": True},
    "inpaint":    {"title": "Inpaint",          "outputs": [("image", "image")],
                   "extra_source_ports": ("images", "layers"),
                   "static": ["image", "mask"], "network": True},
    "slice-3d": {"title": "Slice 3D", "static": ["model", "paletteImage"],
                "outputs": [("sheet", "image"), ("layout", "text"), ("image", "image")]},
    "sprite-stack-render": {"title": "Sprite stack render", "static": ["sheet"],
                            "outputs": [("image", "image"), ("video", "video")]},
    "resize":     {"title": "Resize / crop",    "outputs": [("image", "image")],
                   "static": ["image"]},
    "vision":     {"title": "Vision",           "outputs": [("text", "text")],
                   "static": ["image"], "network": True},
    # ⚖️ Decide: text in + img1… candidates (IMG_PORT_RE); text out (the answer) + image out
    # (pick: the winner; otherwise the first wired image, passed through)
    "decide":     {"title": "Decide",           "outputs": [("text", "text"), ("image", "image")],
                   "extra_source_ports": ("decision", "yes", "no"),
                   "static": ["text"], "dynamic": [IMG_PORT_RE], "network": True},
    "tvideo":     {"title": "Text→Video",       "outputs": [("video", "video")],
                   "dynamic": [REF_PORT_RE], "network": True},
    "ivideo":     {"title": "Image→Video",      "outputs": [("video", "video")],
                   "static": ["image", "endframe"], "network": True},
    # 🧊 3D: image + model ports are static; prompt is a field override. Output is a GLB.
    "model3d":    {"title": "3D model",         "outputs": [("model", "model3d")],
                   "static": ["image", "model"], "network": True},
    "vedit":      {"title": "Video edit",       "outputs": [("video", "video")],
                   "static": ["video"], "network": True},
    "vframes":    {"title": "Video → frames",   "outputs": [("frame1", "image")],
                   "static": ["video"]},
    "combine":    {"title": "Combine videos",   "outputs": [("video", "video")],
                   "dynamic": [VID_PORT_RE, CLIP_PORT_RE]},
    "soundtrack": {"title": "Soundtrack",       "outputs": [("video", "video")],
                   "static": ["video", "audio"]},
    "lipsync":    {"title": "Avatar / lipsync", "outputs": [("video", "video")],
                   "static": ["image", "audio"], "network": True},
    "music":      {"title": "Music",            "outputs": [("audio", "audio")], "network": True},
    "remix":      {"title": "Remix audio",      "outputs": [("audio", "audio")],
                   "static": ["audio"], "network": True},
    "tts":        {"title": "Speech",           "outputs": [("audio", "audio")], "network": True},
    "trim":       {"title": "Trim audio",       "outputs": [("audio", "audio")],
                   "static": ["audio"]},
    "extractaudio": {"title": "Extract audio",  "outputs": [("audio", "audio")],
                     "static": ["video"]},
    "transcribe": {"title": "Transcribe",       "outputs": [("text", "text")],
                   "static": ["audio"], "network": True},
    # 🎧 Clean voice: audio OR video in, cleaned audio out. NanoGPT downloads a public URL.
    "cleanvoice": {"title": "Clean voice",      "outputs": [("audio", "audio")],
                   "static": ["audio", "video"], "network": True},
    # 🔌 Custom endpoint: url/mode stay in the input dict (a Choice path joins the host).
    # Not a NanoGPT call — network is false so an endpoint-only graph needs no API key.
    "endpoint":   {"title": "Custom endpoint",  "outputs": [("text", "text")],
                   "extra_source_ports": ("image", "images", "video", "audio"),
                   "static": ["text", "image", "audio", "video", "url", "mode"]},
    "comment":    {"title": "Comment",          "outputs": [], "note": True},
}

# Field overrides consumed by the SDK runners. Keep their wireability independent
# of stored values and modes: for example a Choice can set a downstream model.
_LORA_FIELDS = ("loras", "loraUrl", "loraStrength")
_IMAGE_FIELDS = ("model", "prompt", "size", "seed", "modelOpts", "customCivitaiAir", *_LORA_FIELDS)
_VIDEO_FIELDS = ("model", "prompt", "resolution", "aspect", "duration", "modelOpts")
_AUDIO_FIELDS = ("model", "prompt", "response_format", "extraJson")
_WIRE_FIELDS = {
    "text": ("text",),
    "upload": ("image", "optional"),
    "aupload": ("audio", "optional"),
    "vupload": ("video", "optional"),
    "mupload": ("model", "optional"),
    "choice": ("options", "selected"),
    "join": ("sep",),
    "llm": ("model", "prompt", "system", "temperature", "maxTokens", "format",
            "reasoningEffort", "showThinking"),
    "image": (*_IMAGE_FIELDS, "variations", "sel"),
    "edit": _IMAGE_FIELDS,
    "inpaint": _IMAGE_FIELDS,
    "slice-3d": ("preset", "clip", "frames", "slices", "resolution", "palette", "outline", "upAxis"),
    "sprite-stack-render": ("yaw", "pitch", "spacing", "scale", "turntable", "fps", "duration",
                            "yawStart", "yawEnd"),
    "resize": ("mode", "width", "height"),
    "vision": ("model", "q"),
    "decide": ("model", "mode", "question", "options", "levels", "gate"),  # branching is authored
    "tvideo": (*_VIDEO_FIELDS, *_LORA_FIELDS),
    "ivideo": (*_VIDEO_FIELDS, *_LORA_FIELDS),
    "vedit": (*_VIDEO_FIELDS, *_LORA_FIELDS),
    "lipsync": _VIDEO_FIELDS,
    "model3d": _VIDEO_FIELDS,
    "vframes": ("frames", "gap", "dir", "stride", "startSec", "endSec"),
    "combine": ("dedup",),
    "soundtrack": ("loop",),
    "music": (*_AUDIO_FIELDS, "lyrics", "instrumental", "duration", "negative_prompt", "seed"),
    "remix": (*_AUDIO_FIELDS, "lyrics", "duration"),
    "tts": (*_AUDIO_FIELDS, "voice", "speed", "instructions"),
    "trim": ("start", "length"),
    "extractaudio": ("start", "length"),
    "transcribe": ("model", "language"),
    "cleanvoice": ("model", "url"),
    "endpoint": ("model", "auth", "prompt", "system", "size"),
}

# Kept for import compatibility; empty — local media ops are implemented (need ffmpeg).
UNSUPPORTED_TYPES = tuple(t for t, s in NODE_TYPES.items() if s.get("unsupported"))


class Node(object):
    __slots__ = ("id", "type", "fields", "name")

    def __init__(self, id, type, fields, name=None):
        self.id = id
        self.type = type
        self.fields = fields
        self.name = name

    def spec(self):
        return NODE_TYPES.get(self.type, {})


class Link(object):
    __slots__ = ("id", "from_node", "from_port", "to_node", "to_port")

    def __init__(self, id, from_node, from_port, to_node, to_port):
        self.id = id
        self.from_node = from_node
        self.from_port = from_port
        self.to_node = to_node
        self.to_port = to_port


class Graph(object):
    def __init__(self, nodes, links, warnings):
        self.nodes = nodes          # ordered dict id -> Node
        self.links = links          # list of Link
        self.warnings = warnings    # list of str

    def node(self, node_id):
        try:
            return self.nodes.get(node_id)
        except TypeError:
            return None

    def valid_links(self):
        """Links that count for inspection and estimation. Run validates every link."""
        return (link for link in self.links if not wire_issue(self, link))

    def inbound(self, node_id):
        return [l for l in self.links if l.to_node == node_id and not wire_issue(self, l)]

    def outbound(self, node_id):
        return [l for l in self.links if l.from_node == node_id and not wire_issue(self, l)]

    def port_is_fed(self, node_id, port):
        return any(l.to_node == node_id and l.to_port == port and not wire_issue(self, l)
                   for l in self.links)


def optional_node(node):
    """Author-marked optional node: the editor's "optional" checkbox on an input node.

    ``fields.optional`` makes every input the node surfaces skippable — the run proceeds
    and the node yields an empty value instead of failing. It is serialized inside fields,
    so it survives save / share / materialize with zero format changes. Mirrors
    nanoodle-js graph.mjs optionalNode.
    """
    if node is None:
        return False
    fields = node.get("fields") if isinstance(node, dict) else getattr(node, "fields", None)
    v = (fields or {}).get("optional")
    return v is True or v == "true"


def display_name(node):
    """node.name (trimmed) -> type title -> type -> '?' (play.html displayName)."""
    if node is None:
        return "?"
    if node.name and str(node.name).strip():
        return str(node.name).strip()
    spec = NODE_TYPES.get(node.type)
    if spec and spec.get("title"):
        return spec["title"]
    return node.type or "?"


def _wire_path(link):
    return "wire %s.%s → %s.%s" % (link.from_node, link.from_port, link.to_node, link.to_port)


def is_input_port(node, port):
    spec = node.spec()
    return isinstance(port, str) and (port in spec.get("static", ())
                                      or any(rx.fullmatch(port) for rx in spec.get("dynamic", ())))


def source_wire_issue(source, link):
    """Describe an undeclared source port, independent of what a run emits."""
    if source is None or source.type not in NODE_TYPES:
        return None
    spec = source.spec()
    port = link.from_port
    if (any(name == port for name, _ in spec["outputs"])
            or port in spec.get("extra_source_ports", ())
            or (source.type == "vframes" and isinstance(port, str) and port in _FRAME_PORT_INDEX)):
        return None
    return ('%s uses undeclared source port "%s" on %s'
            % (_wire_path(link), port, display_name(source)))


def wire_issue(graph, link):
    """Describe the first structural problem with a wire kept for inspection."""
    source, target = graph.node(link.from_node), graph.node(link.to_node)
    if source is None:
        return '%s references missing source node "%s"' % (_wire_path(link), link.from_node)
    if target is None:
        return '%s references missing target node "%s"' % (_wire_path(link), link.to_node)
    issue = source_wire_issue(source, link)
    if issue:
        return issue
    port = link.to_port
    # The editor can attach these media inputs, but Python would discard them.
    video_edit_image = target.type == "vedit" and (
        port == "image" or (isinstance(port, str) and REF_PORT_RE.fullmatch(port)))
    if video_edit_image or (target.type == "llm" and port == "video"):
        return ('%s uses %s input "%s", which the Python SDK cannot send. '
                'Run the graph in the browser or remove the wire.'
                % (_wire_path(link), display_name(target), port))
    if (target.type in NODE_TYPES and not is_input_port(target, port)
            and port not in _WIRE_FIELDS.get(target.type, ())):
        return ('%s uses undeclared target port "%s" on %s'
                % (_wire_path(link), port, display_name(target)))
    return None


# Media fields hold a data: or http(s) URL (the editor writes data: URLs; hand-authored
# graphs may inline either). Agents sometimes leave prose placeholders ("[image will be
# provided at run time]") or bare file paths — those LOOK filled (inspect shows a default,
# the required-input check passes) but would post garbage to the API. Treat anything that
# isn't a real media URL as EMPTY at load, and say how to actually supply the media.
_MEDIA_FIELD_KEYS = ("image", "mask", "audio", "video")
_MEDIA_URL_RE = re.compile(r"^\s*(data:|https?:)", re.I)


def _scrub_media_placeholders(nid, fields, warnings, ntype=None):
    # ``model`` is a model id on every generative node. Only the 3D upload stores
    # a file URL under that key, so only that node scrubs a prose placeholder.
    keys = _MEDIA_FIELD_KEYS + (("model",) if ntype == "mupload" else ())
    for key in keys:
        v = fields.get(key)
        # The editor serializes its built-in samples relative to its own origin.
        if isinstance(v, str) and re.fullmatch(r"examples/[A-Za-z0-9_./-]+", v) and ".." not in v:
            fields[key] = "https://nanoodle.com/" + v
            continue
        if v is None or v == "" or (isinstance(v, str) and _MEDIA_URL_RE.match(v)):
            continue
        if isinstance(v, str):
            shown = '"%s"' % (v if len(v) <= 60 else v[:57] + "…")
        else:
            shown = "a non-string value (%s)" % type(v).__name__
        fields[key] = ""
        warnings.append(
            "node %s: fields.%s held %s — not a data: or http(s) URL, so it was treated as empty. "
            "Leave media fields \"\" in the graph and supply the %s at run time "
            "(CLI: --input \"<key>=@file\", library: run({\"<key>\": media_from_file(path)}), "
            "key from inspect/wf.inputs), or inline a data: URL." % (nid, key, shown, key))


def materialize(data, warnings=None):
    """Build a Graph from parsed noodle-graph.json (or a minimal {nodes, links}).

    Loader semantics replicated from the app:
      - type alias: audio -> tts (legacy)
      - links into a music/tts node's "text" port are migrated to "prompt"
      - wires with missing nodes or undeclared ports are kept with a warning
    Unknown node types are KEPT (with a load warning) — run() fails fast on them.
    """
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, dict):
        raise NanoodleError("workflow JSON must be an object with a 'nodes' list")
    warnings = warnings if warnings is not None else []

    nodes = {}
    for raw in (data.get("nodes") or []):
        ntype = raw.get("type")
        if ntype == "audio":  # legacy alias
            ntype = "tts"
        if ntype == "decide-branch":
            ntype = "decide"
        nid = raw.get("id")
        if not nid:
            warnings.append("skipped a node with no id")
            continue
        if nid in nodes:
            # twin of JS graph.mjs: a duplicate id is a load error, never a
            # silent overwrite of the first node (D4, fixed 2026-11-18)
            raise NanoodleError('duplicate node id "%s" — every node must have a unique id' % nid)
        if ntype not in NODE_TYPES:
            warnings.append("unknown node type %r (node %s) — it cannot be executed" % (ntype, nid))
        fields = dict(raw.get("fields") or {})
        if raw.get("type") == "decide-branch":
            fields["branch"] = True
        if ntype in NODE_TYPES:
            _scrub_media_placeholders(nid, fields, warnings, ntype)
        nodes[nid] = Node(nid, ntype, fields, raw.get("name"))

    graph = Graph(nodes, [], warnings)
    for raw in (data.get("links") or []):
        raw = raw if isinstance(raw, dict) else {}
        f, t = raw.get("from"), raw.get("to")
        f = f if isinstance(f, dict) else {}
        t = t if isinstance(t, dict) else {}
        fn, fp, tn, tp = f.get("node"), f.get("port"), t.get("node"), t.get("port")
        # migration: legacy music/tts inbound "text" port is now "prompt"
        target = graph.node(tn)
        if tp == "text" and target is not None and target.type in ("music", "tts"):
            tp = "prompt"
        link = Link(raw.get("id"), fn, fp, tn, tp)
        graph.links.append(link)
        issue = wire_issue(graph, link)
        if issue:
            warnings.append(issue)

    return graph


def topo_order(graph):
    """Kahn topological order over non-comment nodes; cycles raise naming the nodes."""
    ids = [nid for nid, n in graph.nodes.items() if not n.spec().get("note")]
    idset = set(ids)
    indeg = {nid: 0 for nid in ids}
    down = {nid: [] for nid in ids}
    seen_edges = set()
    for l in graph.valid_links():
        if l.from_node in idset and l.to_node in idset:
            edge = (l.from_node, l.to_node)
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            indeg[l.to_node] += 1
            down[l.from_node].append(l.to_node)
    queue = [nid for nid in ids if indeg[nid] == 0]
    order = []
    while queue:
        nid = queue.pop(0)
        order.append(nid)
        for d in down[nid]:
            indeg[d] -= 1
            if indeg[d] == 0:
                queue.append(d)
    if len(order) != len(ids):
        cyclic = sorted(nid for nid in ids if nid not in order)
        names = ", ".join("%s (%s)" % (nid, display_name(graph.node(nid))) for nid in cyclic)
        raise NanoodleError("workflow has a cycle involving: %s" % names)
    return order


def classify_inbound(node, links):
    """Split a node's inbound links into media/text INPUT ports vs FIELD OVERRIDES.

    Declared static ports and dynamic families feed the node's input dict;
    A declared field wire overrides the same-named field (that is how wired
    prompt/system/lyrics/q replace typed values). Returns (inputs, overrides)
    as {port_or_field: (from_node, from_port)}.
    """
    inputs, overrides = {}, {}
    for l in links:
        port = l.to_port
        if is_input_port(node, port):
            inputs[port] = (l.from_node, l.from_port)
        elif port in _WIRE_FIELDS.get(node.type, ()):
            overrides[port] = (l.from_node, l.from_port)
    return inputs, overrides
