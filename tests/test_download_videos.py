"""scripts/download_videos.py without the network: URL/MD5 mapping, ranged download, resume."""
import csv
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest

import download_videos as dv


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


@pytest.fixture
def fake_repo(tmp_path):
    """The four data files of the official EPIC downloader, with made-up rows."""
    data = tmp_path / "repo" / "data"
    write_csv(data / "epic_100_splits.csv", ["video_id", "ar_val"], [["P03_26", "TRUE"], ["P01_101", "TRUE"]])
    write_csv(data / "epic_55_splits.csv", ["participant_id", "video_id", "split"], [["P03", "P03_26", "test"]])
    write_csv(data / "md5.csv", ["md5", "file_remote_path", "version"],
              [["aaa", "videos/test/P03/P03_26.MP4", "55"], ["bbb", "P01/videos/P01_101.MP4", "100"],
               ["ccc", "videos/test/P03/P03_26.tar", "55"]])
    write_csv(data / "errata.csv", ["rdsf_path", "dropbox_path"], [["P01/videos/P01_101.MP4", "https://fix/P01_101"]])
    return tmp_path / "repo"


def test_epic_sources_follow_the_official_layout(fake_repo):
    sources = dv.epic_sources(fake_repo, ["P03_26", "P01_101"])
    assert sources["P03_26"] == (f"{dv.EPIC_55_URL}/videos/test/P03/P03_26.MP4", "aaa")
    assert sources["P01_101"] == ("https://fix/P01_101", None)   # erratum wins; no errata MD5 listed
    with pytest.raises(SystemExit, match="P99_01"):
        dv.epic_sources(fake_repo, ["P99_01"])


def test_fpvsum_ids_come_from_the_upstream_list(tmp_path):
    cache = tmp_path / "list.csv"
    write_csv(cache, ["video_name", "youtube_id", "category", "split"],
              [["BikePolo_DHbYfmjX3WQ_1", "hkNp3udSNV8", "Bikepolo", "labeled"],
               ["Bike_Q_maV87d1I8", "Q_maV87d1I8", "Bike", "labeled"]])
    ids = dv.fpvsum_youtube_ids(["BikePolo_DHbYfmjX3WQ_1", "Bike_Q_maV87d1I8"], cache)
    assert ids == {"BikePolo_DHbYfmjX3WQ_1": "hkNp3udSNV8", "Bike_Q_maV87d1I8": "Q_maV87d1I8"}
    with pytest.raises(SystemExit, match="Ski_x"):
        dv.fpvsum_youtube_ids(["Ski_x"], cache)


class RangeServer:
    """Local HTTP server for one blob, answering HEAD and Range GETs with 206."""

    def __init__(self, blob, fail_first=0):
        self.blob, self.requests, self.fail_left = blob, [], fail_first
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_HEAD(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(owner.blob)))
                self.end_headers()

            def do_GET(self):
                owner.requests.append(self.headers["Range"])
                if owner.fail_left > 0:
                    owner.fail_left -= 1
                    self.send_error(503)
                    return
                start, end = map(int, self.headers["Range"].split("=")[1].split("-"))
                body = owner.blob[start:end + 1]
                self.send_response(206)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v.MP4"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def small_chunks(monkeypatch):
    monkeypatch.setattr(dv, "CHUNK", 1000)
    monkeypatch.setattr(dv.time, "sleep", lambda s: None)


def run_all(download):
    stop = threading.Event()
    for i in list(download.chunks):
        download.fetch(i, stop)


def test_ranged_download_checks_md5_and_resumes(tmp_path, small_chunks):
    blob = bytes(range(256)) * 20   # 5120 bytes = 6 chunks
    server = RangeServer(blob, fail_first=2)   # first two requests fail, then retry succeeds
    try:
        path = tmp_path / "P03" / "videos" / "P03_26.MP4"
        md5 = hashlib.md5(blob).hexdigest()

        first = dv.RangedDownload("P03_26", server.url, md5, path)
        assert len(first.chunks) == 6
        stop = threading.Event()
        for i in first.chunks[:3]:   # interrupted halfway
            first.fetch(i, stop)
        assert not path.exists() and json.loads(first.state.read_text())["done"] == [0, 1, 2]

        resumed = dv.RangedDownload("P03_26", server.url, md5, path)
        assert resumed.chunks == [3, 4, 5]
        run_all(resumed)
        assert path.read_bytes() == blob and not resumed.part.exists() and not resumed.state.exists()
        assert len(server.requests) == 6 + 2
    finally:
        server.close()


def test_md5_mismatch_deletes_the_file(tmp_path, small_chunks):
    server = RangeServer(b"x" * 2500)
    try:
        path = tmp_path / "P01_11.MP4"
        download = dv.RangedDownload("P01_11", server.url, "0" * 32, path)
        run_all(download)
        assert not path.exists() and not download.part.exists() and not download.state.exists()
    finally:
        server.close()


def test_partial_or_wrong_files_count_as_not_downloaded(media, tmp_path):
    import video_io
    out = tmp_path / "videos"
    (out / "fpvsum").mkdir(parents=True)
    good = out / "fpvsum" / "good.mp4"
    good.write_bytes(media["short"].read_bytes())
    (out / "fpvsum" / "broken.mp4").write_bytes(media["short"].read_bytes()[:5000])
    duration = video_io.inspect_video(good).duration
    manifest = {"good": (duration, "fpvsum"), "broken": (duration, "fpvsum"),
                "wrong_length": (duration + 30, "fpvsum"), "absent": (10.0, "fpvsum")}
    (out / "fpvsum" / "wrong_length.mp4").write_bytes(media["short"].read_bytes())
    assert dv.still_needed(manifest, out, list(manifest)) == ["broken", "wrong_length", "absent"]
    report = dv.check(manifest, out, ["fpvsum"])
    assert report["summary"] == {"ok": 1, "unreadable": 1, "duration_mismatch": 1, "missing": 1}


def test_parts_split_each_dataset_evenly_and_cover_everything():
    manifest = {f"P01_{i:02d}": (float(i), "epic_kitchens_100") for i in range(1, 11)}
    manifest.update({f"Ski_{i}": (float(i), "fpvsum") for i in range(4)})
    shares = [dv.share(manifest, f"{k}/3") for k in (1, 2, 3)]
    assert sorted(v for s in shares for v in s) == sorted(manifest)
    assert [len(s) for s in shares] == [4 + 2, 3 + 1, 3 + 1]
    with pytest.raises(SystemExit):
        dv.share(manifest, "4/3")


def test_resume_state_from_another_chunk_size_starts_over(tmp_path, small_chunks, monkeypatch):
    server = RangeServer(b"y" * 3000)
    try:
        path = tmp_path / "P01_12.MP4"
        first = dv.RangedDownload("P01_12", server.url, None, path)
        first.fetch(0, threading.Event())
        monkeypatch.setattr(dv, "CHUNK", 500)
        again = dv.RangedDownload("P01_12", server.url, None, path)
        assert again.chunks == list(range(6))
        run_all(again)
        assert path.read_bytes() == b"y" * 3000
    finally:
        server.close()
