"""Execute the release's shared pixel renderer in a private, in-process QuickJS realm."""
import base64
import json
import struct
import tempfile
import zlib
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

from .errors import NanoodleError
from .local_media import _run, _url_bytes
from .media import make_data_url, sniff_mime


@lru_cache(maxsize=1)
def _program():
    return files(__package__).joinpath("sprite-core.js").read_text(encoding="utf-8")


def _base64(data):
    return base64.b64encode(data).decode("ascii")


def _png(image):
    width, height = image["width"], image["height"]
    raw = base64.b64decode(image["rgba"], validate=True)
    stride = width * 4
    if len(raw) != stride * height or width < 1 or height < 1:
        raise NanoodleError("Invalid sprite frame dimensions")

    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    rows = b"".join(b"\0" + raw[i:i+stride] for i in range(0, len(raw), stride))
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def _image(encoded, directory, ffmpeg):
    raw = base64.b64decode(encoded, validate=True)
    mime = sniff_mime(raw)
    if not mime.startswith("image/"):
        raise NanoodleError("Expected a PNG, JPEG, WebP or GIF image")
    source, output = directory / ("source." + mime.split("/")[1]), directory / "decoded.pam"
    source.write_bytes(raw)
    ffmpeg(["-protocol_whitelist", "file", "-f", "image2", "-i", str(source),
            "-frames:v", "1", "-c:v", "pam", "-pix_fmt", "rgba", str(output)])
    header, raw = output.read_bytes().split(b"ENDHDR\n", 1)
    meta = dict(line.split(None, 1) for line in header.splitlines()[1:] if line and not line.startswith(b"#"))
    width, height = int(meta[b"WIDTH"]), int(meta[b"HEIGHT"])
    if int(meta[b"DEPTH"]) != 4 or len(raw) != width * height * 4:
        raise NanoodleError("Could not decode sprite image as RGBA")
    return json.dumps({"width": width, "height": height, "rgba": _base64(raw)})


def run_sprite(engine, node, inp, on_cost):
    # Imported only by these two local nodes. Each worker thread owns its realm.
    import quickjs
    opts = engine._local_opts()
    source_port = "model" if node.type == "slice-3d" else "sheet"
    if not inp.get(source_port):
        raise NanoodleError(f"Wire {source_port} into {node.type}.")
    source = _base64(_url_bytes(inp[source_port], opts["fetcher"]))
    fields = node.fields
    use_palette = fields.get("palette", 16) not in (0, "0")
    palette = _base64(_url_bytes(inp["paletteImage"], opts["fetcher"])) if use_palette and inp.get("paletteImage") else ""
    engine.check_cancel()

    def ffmpeg(args):
        _run("ffmpeg", ["-nostdin", "-y", "-v", "error", *args],
             cancel_check=opts["cancel_check"], deadline=opts["deadline"])

    with tempfile.TemporaryDirectory(prefix="nanoodle-sprite-") as tmp:
        directory = Path(tmp)
        realm = quickjs.Context()
        realm.add_callable("nativeUtf8", lambda b: base64.b64decode(b).decode("utf-8", errors="replace"))
        realm.add_callable("nativeEncodeUtf8", lambda text: _base64(text.encode("utf-8")))

        def read_image(encoded):
            try:
                return _image(encoded, directory, ffmpeg)
            except Exception as exc:
                return json.dumps({"error": str(exc)})
        realm.add_callable("nativeImage", read_image)
        realm.eval(_program())
        realm.get("startSprite")(node.type, json.dumps(fields), source, palette)
        while realm.execute_pending_job():
            engine.check_cancel()
        result = json.loads(realm.eval("spriteResult"))
        if "error" in result:
            raise NanoodleError(result["error"])
        out = result["out"]
        images = {key: engine._media_ref(make_data_url(_png(out[key]), "image/png"))
                  for key in ("sheet", "image") if key in out}
        if "layout" in out:
            images["layout"] = out["layout"]
        plan = out.get("plan")
        if plan:
            frame = realm.get("spriteFrame")
            for index in range(plan["count"]):
                engine.check_cancel()
                (directory / f"{index:05}.png").write_bytes(_png(json.loads(frame(index))))
            target = directory / "turntable.mp4"
            args = ["-framerate", str(plan["count"] / plan["duration"]), "-i", str(directory / "%05d.png"), "-an"]
            try:
                ffmpeg([*args, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)])
            except NanoodleError as exc:
                if "unknown encoder" not in str(exc).lower() or "libx264" not in str(exc).lower():
                    raise
                ffmpeg([*args, "-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)])
            images["video"] = engine._media_ref(make_data_url(target.read_bytes(), "video/mp4"))
        return images
