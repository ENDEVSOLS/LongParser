"""Abstracted job queue backend for LongParser.

ARQ (async Redis queue) is the default implementation.
The QueueBackend ABC allows swapping to Celery/Dramatiq
without rewriting business logic.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger(__name__)


class QueueBackend(ABC):
    """Abstract queue interface — swap implementations without changing routes."""

    @abstractmethod
    async def enqueue(self, task_name: str, payload: dict) -> str:
        """Enqueue a task. Returns task/job reference ID."""
        ...

    @abstractmethod
    async def cancel(self, task_id: str) -> bool:
        """Cancel a queued/running task. Returns True if cancelled."""
        ...

    @abstractmethod
    async def status(self, task_id: str) -> dict:
        """Get task status and progress."""
        ...


class ARQBackend(QueueBackend):
    """ARQ (async Redis queue) implementation."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self._pool = None

    async def _get_pool(self):
        if self._pool is None:
            from arq import create_pool
            from arq.connections import RedisSettings

            self._pool = await create_pool(RedisSettings.from_dsn(self.redis_url))
        return self._pool

    async def enqueue(self, task_name: str, payload: dict) -> str:
        """Enqueue a task via ARQ."""
        pool = await self._get_pool()
        job = await pool.enqueue_job(task_name, **payload)
        job_id = job.job_id if job else "unknown"
        logger.info(f"Enqueued {task_name} → ARQ job {job_id}")
        return job_id

    async def cancel(self, task_id: str) -> bool:
        """Cancel via ARQ job abort (best-effort)."""
        try:
            from arq.jobs import Job
            pool = await self._get_pool()
            job = Job(task_id, pool)
            await job.abort()
            return True
        except Exception as e:
            logger.warning(f"Failed to abort ARQ job {task_id}: {e}")
            return False

    async def status(self, task_id: str) -> dict:
        """Get ARQ job status."""
        try:
            from arq.jobs import Job
            pool = await self._get_pool()
            job = Job(task_id, pool)
            info = await job.info()
            if info is None:
                return {"status": "unknown"}
            return {
                "status": info.status,
                "result": info.result,
                "enqueue_time": str(info.enqueue_time) if info.enqueue_time else None,
            }
        except Exception:
            return {"status": "unknown"}

    async def close(self) -> None:
        """Close the Redis pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
