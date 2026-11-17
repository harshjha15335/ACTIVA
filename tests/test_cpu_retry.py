"""D23: a CUDA out-of-memory error in a stage retries that stage once on the CPU."""
import sys
import types

import pytest

import config
import main


class OutOfMemoryError(RuntimeError):  # same class name as torch.cuda.OutOfMemoryError
    pass


@pytest.fixture
def on_cuda(monkeypatch):
    monkeypatch.setenv("ACTIVA_DEVICE", "cuda")
    monkeypatch.setattr(config, "DEVICE", "cuda")
    fake_actions = types.SimpleNamespace(_ENGINE={"slowfast": "on-gpu"})
    monkeypatch.setitem(sys.modules, "actions", fake_actions)
    return fake_actions


def run_stage(compute, fallback=lambda: "fallback"):
    timings, fallbacks = {}, []
    value = main._stage("actions", compute, lambda v: None, fallback, timings, fallbacks)
    return value, fallbacks


def test_oom_retries_once_on_cpu(on_cuda):
    devices = []

    def compute():
        devices.append(config.DEVICE)
        if config.DEVICE != "cpu":
            raise OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return "real actions"

    value, fallbacks = run_stage(compute)
    assert value == "real actions" and fallbacks == []
    assert devices == ["cuda", "cpu"]
    assert config.device() == "cpu" and on_cuda._ENGINE == {}


def test_wrapped_oom_message_also_retries(on_cuda):
    calls = []

    def compute():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")
        return "ok"

    assert run_stage(compute) == ("ok", [])


def test_other_errors_do_not_retry(on_cuda):
    calls = []

    def compute():
        calls.append(1)
        raise ValueError("bad video")

    assert run_stage(compute) == ("fallback", ["actions"])
    assert len(calls) == 1 and config.DEVICE == "cuda" and on_cuda._ENGINE


def test_oom_on_cpu_is_not_retried(on_cuda, monkeypatch):
    monkeypatch.setenv("ACTIVA_DEVICE", "cpu")
    calls = []

    def compute():
        calls.append(1)
        raise OutOfMemoryError("out of memory")

    assert run_stage(compute) == ("fallback", ["actions"])
    assert len(calls) == 1


def test_second_oom_falls_back(on_cuda):
    calls = []

    def compute():
        calls.append(1)
        raise OutOfMemoryError("CUDA out of memory")

    assert run_stage(compute) == ("fallback", ["actions"])
    assert len(calls) == 2


def test_strict_mode_still_fails_when_the_cpu_retry_fails(on_cuda, monkeypatch):
    monkeypatch.setenv("ACTIVA_STRICT", "1")

    def compute():
        raise OutOfMemoryError("CUDA out of memory")

    with pytest.raises(main.StageError):
        run_stage(compute)
