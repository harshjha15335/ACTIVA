"""scripts/setup_weights.py against a local HTTP server with fake assets (no internet)."""
import functools
import hashlib
import http.server
import importlib.util
import io
from pathlib import Path
import threading
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("setup_weights", ROOT / "scripts" / "setup_weights.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _zip(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        for name, data in entries.items():
            z.writestr(name, data)
    return buffer.getvalue()


@pytest.fixture
def release(tmp_path, monkeypatch):
    """A fake release served over HTTP, and setup_weights pointed at a fake repo root."""
    sw = _load()
    served, repo = tmp_path / "served", tmp_path / "repo"
    served.mkdir()
    blobs = {"clip.safetensors": b"clip" * 1000, "clip.manifest.json": b"{}", "slowfast.pyth": b"sf" * 5000}
    assets = {}
    for name, data in blobs.items():
        (served / name).write_bytes(data)
        assets[name] = (f"weights/{name}", len(data), hashlib.sha256(data).hexdigest())
    code = _zip({marker: b"x" for marker in ["a/config.yaml", "classes.csv"]})
    (served / "code.zip").write_bytes(code)
    monkeypatch.setattr(sw, "ROOT", repo)
    monkeypatch.setattr(sw, "ASSETS", assets)
    monkeypatch.setattr(sw, "CODE_ZIP", ("code.zip", "weights/m1", hashlib.sha256(code).hexdigest()))
    monkeypatch.setattr(sw, "CODE_MARKERS", ["a/config.yaml", "classes.csv"])

    requests = []

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            requests.append(self.path)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(served)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield sw, f"http://127.0.0.1:{server.server_address[1]}", repo, served, requests
    server.shutdown()


def test_fresh_install_then_rerun_downloads_nothing(release):
    sw, url, repo, _, requests = release
    assert sw.main(["--base-url", url]) == 0
    assert (repo / "weights/slowfast.pyth").read_bytes() == b"sf" * 5000
    assert (repo / "weights/m1/a/config.yaml").is_file()
    assert sw.main(["--check"]) == 0
    before = len(requests)
    assert sw.main(["--base-url", url]) == 0
    assert len(requests) == before, "verified files must not be downloaded again"


def test_wrong_local_file_is_replaced(release):
    sw, url, repo, _, _ = release
    (repo / "weights").mkdir(parents=True)
    (repo / "weights/slowfast.pyth").write_bytes(b"old weights")
    assert sw.main(["--check"]) == 1
    assert sw.main(["--base-url", url]) == 0
    assert (repo / "weights/slowfast.pyth").read_bytes() == b"sf" * 5000


def test_corrupt_download_is_deleted_and_fails(release):
    sw, url, repo, served, _ = release
    (served / "slowfast.pyth").write_bytes(b"tampered")
    with pytest.raises(SystemExit, match="SHA-256 mismatch"):
        sw.main(["--base-url", url])
    assert not (repo / "weights/slowfast.pyth").exists()


def test_zip_with_escaping_path_is_refused(release, monkeypatch):
    sw, url, _, served, _ = release
    evil = _zip({"../../outside.txt": b"x"})
    (served / "code.zip").write_bytes(evil)
    monkeypatch.setattr(sw, "CODE_ZIP", ("code.zip", "weights/m1", hashlib.sha256(evil).hexdigest()))
    with pytest.raises(SystemExit, match="unsafe path"):
        sw.main(["--base-url", url])


def test_real_asset_table_matches_the_m1_pins():
    sw = _load()
    m1 = importlib.util.spec_from_file_location("setup_m1_assets", ROOT / "scripts" / "setup_m1_assets.py")
    module = importlib.util.module_from_spec(m1)
    m1.loader.exec_module(module)
    ours = {dest.removeprefix("weights/m1/"): want for dest, _, want in sw.ASSETS.values()}
    for name, want in module.EXPECTED_SHA256.items():
        assert ours[name] == want
