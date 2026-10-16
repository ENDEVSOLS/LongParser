"""Queue lifecycle and failure handling using only controlled Redis/ARQ SDKs."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from longparser.server.queue import ARQBackend


@pytest.fixture
def queue_sdk(monkeypatch):
    import arq
    import arq.jobs

    pool = SimpleNamespace(enqueue_job=AsyncMock(return_value=SimpleNamespace(job_id="task-1")),
                           close=AsyncMock())
    create_pool = AsyncMock(return_value=pool)
    job = SimpleNamespace(abort=AsyncMock(), info=AsyncMock())
    job_factory = Mock(return_value=job)
    monkeypatch.setattr(arq, "create_pool", create_pool)
    monkeypatch.setattr(arq.jobs, "Job", job_factory)
    return pool, create_pool, job, job_factory


async def test_enqueue_reuses_pool_and_closes_it(queue_sdk):
    pool, factory, _, _ = queue_sdk
    backend = ARQBackend("redis://localhost:6379/4")
    assert await backend.enqueue("extract", {"job_id": "j", "tenant_id": "t"}) == "task-1"
    pool.enqueue_job.assert_awaited_once_with("extract", job_id="j", tenant_id="t")
    pool.enqueue_job.return_value = None
    assert await backend.enqueue("extract", {}) == "unknown"
    factory.assert_awaited_once()
    await backend.close()
    pool.close.assert_awaited_once()
    await backend.close()
    assert pool.close.await_count == 1


async def test_cancel_success_and_failure_are_explicit(queue_sdk):
    _, _, job, factory = queue_sdk
    backend = ARQBackend()
    assert await backend.cancel("task") is True
    factory.assert_called_once()
    job.abort.assert_awaited_once()
    job.abort.side_effect = RuntimeError("cannot abort")
    assert await backend.cancel("task") is False


async def test_status_formats_timestamp_and_handles_unknown_jobs(queue_sdk):
    _, _, job, _ = queue_sdk
    backend = ARQBackend()
    job.info.return_value = SimpleNamespace(status="complete", result={"pages": 2}, enqueue_time="today")
    assert await backend.status("task") == {"status": "complete", "result": {"pages": 2}, "enqueue_time": "today"}
    job.info.return_value.enqueue_time = None
    assert (await backend.status("task"))["enqueue_time"] is None
    job.info.return_value = None
    assert await backend.status("missing") == {"status": "unknown"}
    job.info.side_effect = OSError("Redis down")
    assert await backend.status("task") == {"status": "unknown"}
