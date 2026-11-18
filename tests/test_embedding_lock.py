"""Concurrency and failure behaviour of the lazy embedding model.

``EmbeddingService.model`` is a lazy property. It was unlocked, which was safe
only while every caller was single-threaded — true until model loading moved off
the startup path (issue #24). Two threads reaching an unpopulated property each
built a ``TextEmbedding``, i.e. two ONNX sessions racing on one cache directory.

These tests pin the behaviour without downloading or loading a real model: the
``fastembed`` import is patched inside the module the property imports from.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from guaipeca.embedding import EmbeddingService

DIMS = 384


class _FakeModel:
    """Stands in for a built fastembed model."""

    def embed(self, texts):
        for _ in texts:
            yield np.zeros(DIMS, dtype=np.float32)


class _CountingBuilder:
    """Records how many models get constructed, with an optional delay.

    ``delay`` widens the window in which a second thread would otherwise see an
    unpopulated ``_model`` and start building its own.
    """

    def __init__(self, delay: float = 0.0, fail: BaseException | None = None) -> None:
        self.calls = 0
        self.delay = delay
        self.fail = fail
        self._lock = threading.Lock()

    def __call__(self, **_kwargs):
        with self._lock:
            self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.fail is not None:
            raise self.fail
        return _FakeModel()


@pytest.fixture
def service():
    return EmbeddingService(model_name="all-MiniLM-L6-v2", dimensions=DIMS)


def _patch_builder(monkeypatch, builder):
    """Point the property's builder at a stub.

    ``_build_model`` imports fastembed lazily, so patching the builder method is
    simpler and more honest than patching a module attribute that only exists
    inside the function body.
    """
    monkeypatch.setattr(EmbeddingService, "_build_model", lambda self: builder())


def test_concurrent_access_builds_once(monkeypatch, service):
    """Four threads racing an unpopulated property must build one model.

    Proven failure before the lock: the same test constructed four models, which
    means four ONNX sessions against one cache directory.
    """
    builder = _CountingBuilder(delay=0.2)
    _patch_builder(monkeypatch, builder)

    results: list[object] = []
    errors: list[BaseException] = []

    def access():
        try:
            results.append(service.model)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=access) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors, f"concurrent access raised: {errors!r}"
    assert builder.calls == 1, (
        f"model was built {builder.calls} times; concurrent callers must wait for "
        "the first build rather than construct their own"
    )
    assert len(results) == 4
    assert all(r is results[0] for r in results), "every caller must see the same instance"


def test_failed_build_is_not_retried(monkeypatch, service):
    """A broken model must fail fast, not rebuild on every call.

    Without a remembered failure, each request would pay the full build cost
    again and log a fresh traceback — a retry storm behind a single defect.
    """
    builder = _CountingBuilder(fail=RuntimeError("model unavailable"))
    _patch_builder(monkeypatch, builder)

    with pytest.raises(RuntimeError, match="model unavailable"):
        _ = service.model
    with pytest.raises(RuntimeError, match="model unavailable"):
        _ = service.model
    with pytest.raises(RuntimeError, match="model unavailable"):
        _ = service.model

    assert builder.calls == 1, (
        f"builder ran {builder.calls} times; a failed build must be remembered, "
        "not retried per call"
    )


def test_failure_is_visible_to_concurrent_waiters(monkeypatch, service):
    """Threads waiting behind a failing build must see the error, not a hang."""
    builder = _CountingBuilder(delay=0.15, fail=RuntimeError("boom"))
    _patch_builder(monkeypatch, builder)

    errors: list[BaseException] = []

    def access():
        try:
            _ = service.model
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=access) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert len(errors) == 3, f"expected every waiter to see the failure, got {errors!r}"
    assert all(isinstance(e, RuntimeError) and "boom" in str(e) for e in errors)


def test_is_loaded_is_false_until_built(monkeypatch, service):
    """``is_loaded`` must not trigger a build; it only reports state."""
    builder = _CountingBuilder()
    _patch_builder(monkeypatch, builder)

    assert service.is_loaded is False
    assert builder.calls == 0, "checking is_loaded must not load the model"

    _ = service.model
    assert service.is_loaded is True
