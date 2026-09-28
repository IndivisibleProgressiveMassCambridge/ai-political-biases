"""Build the run matrix and collect responses. Resumable: jobs already in the store are skipped."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_random_exponential

from biasstudy.config import Study
from biasstudy.providers import Provider, Request
from biasstudy.store import Store

log = logging.getLogger(__name__)

ARMS = ("nosearch", "search")
TEMP_MODES = ("default", "zero")


@dataclass(frozen=True)
class Job:
    model_key: str
    arm: str
    temp_mode: str
    race_id: str
    candidate_id: str | None  # None for race-tier prompts
    prompt_id: str
    rep: int
    prompt_text: str
    candidate_order: tuple[str, ...]  # order candidates appear in race prompts

    @property
    def key(self) -> str:
        return job_key(self.model_key, self.arm, self.temp_mode, self.race_id,
                       self.candidate_id, self.prompt_id, self.rep)


def job_key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:20]


def build_jobs(
    study: Study,
    models: list[str] | None = None,
    races: list[str] | None = None,
    reps: int = 3,
    arms: tuple[str, ...] = ARMS,
    temp_modes: tuple[str, ...] = ("default",),
) -> list[Job]:
    jobs: list[Job] = []
    for spec in study.models:
        # Explicitly named models run even if disabled in config (e.g. a one-off Grok check).
        if (models and spec.key not in models) or (not models and not spec.enabled):
            continue
        for arm in arms:
            if arm == "search" and not spec.supports_search:
                continue
            if arm == "nosearch" and not spec.supports_nosearch:
                continue
            for temp_mode in temp_modes:
                if temp_mode == "zero" and not spec.supports_temperature:
                    continue
                for race in study.races:
                    if races and race.id not in races:
                        continue
                    for prompt in study.prompts:
                        if not prompt.enabled or prompt.style not in spec.styles:
                            continue
                        targets = [None] if prompt.tier == "race" else [c.id for c in race.candidates]
                        for cand_id in targets:
                            for rep in range(reps):
                                key = job_key(spec.key, arm, temp_mode, race.id, cand_id, prompt.id, rep)
                                # Deterministic per-job shuffle so reruns reproduce the exact prompt.
                                order = [c.id for c in race.candidates]
                                random.Random(key).shuffle(order)
                                jobs.append(Job(
                                    model_key=spec.key, arm=arm, temp_mode=temp_mode, race_id=race.id,
                                    candidate_id=cand_id, prompt_id=prompt.id, rep=rep,
                                    prompt_text=render_prompt(study, race.id, prompt.id, cand_id, order),
                                    candidate_order=tuple(order),
                                ))
    return jobs


def render_prompt(study: Study, race_id: str, prompt_id: str, cand_id: str | None, order: list[str]) -> str:
    race = study.race(race_id)
    prompt = study.prompt(prompt_id)
    by_id = {c.id: c for c in race.candidates}
    if prompt.tier == "race":
        a, b = by_id[order[0]], by_id[order[1]]
        return prompt.text.format(
            year=race.year, district_desc=race.district_desc,
            candidate_a=a.name, candidate_b=b.name, search_hint=a.search_hint,
        ).strip()
    cand = by_id[cand_id]
    return prompt.text.format(candidate=cand.name, office_desc=cand.office_desc,
                              search_hint=cand.search_hint, year=race.year).strip()


def _retryable(exc: BaseException) -> bool:
    # Retry rate limits, timeouts, 5xx, connection errors. Don't retry 400/401/404.
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int):
        return status == 408 or status == 409 or status == 429 or status >= 500
    return True


async def run_jobs(
    jobs: list[Job],
    providers: dict[str, Provider],
    store: Store,
    errors: Store,
    concurrency: int = 4,
    study_locations: dict[str, str] | None = None,
) -> dict[str, int]:
    study_locations = study_locations or {}
    done = store.keys()
    pending = [j for j in jobs if j.key not in done]
    log.info("%d jobs total, %d already done, %d to run", len(jobs), len(jobs) - len(pending), len(pending))

    sems: dict[str, asyncio.Semaphore] = defaultdict(lambda: asyncio.Semaphore(concurrency))
    stats = {"ok": 0, "error": 0, "skipped": len(jobs) - len(pending)}

    async def one(job: Job) -> None:
        provider = providers[job.model_key]
        spec = provider.spec
        req = Request(
            prompt=job.prompt_text,
            search=job.arm == "search",
            temperature=0.0 if job.temp_mode == "zero" else None,
            max_output_tokens=spec.max_output_tokens,
            location=study_locations.get(job.race_id, "United States"),
        )
        async with sems[job.model_key]:
            try:
                async for attempt in AsyncRetrying(
                    retry=retry_if_exception(_retryable),
                    wait=wait_random_exponential(multiplier=2, max=120),
                    stop=stop_after_attempt(6),
                    reraise=True,
                ):
                    with attempt:
                        resp = await provider.complete(req)
            except Exception as exc:  # recorded and retried on the next run
                errors.append({"key": job.key, "job": asdict(job), "error": repr(exc),
                               "at": datetime.now(timezone.utc).isoformat()})
                stats["error"] += 1
                log.warning("job %s failed: %r", job.key, exc)
                return
        store.append({
            "key": job.key,
            **asdict(job),
            "request": {**asdict(req), "model_id": spec.model_id, "provider": spec.provider},
            "text": resp.text,
            "shown": resp.shown,
            "citations": resp.citations,
            "model_id_reported": resp.model_id_reported,
            "served_by": resp.served_by,
            "finish_reason": resp.finish_reason,
            "usage": resp.usage,
            "raw": resp.raw,
            "collected_at": datetime.now(timezone.utc).isoformat(),
        })
        stats["ok"] += 1

    await asyncio.gather(*(one(j) for j in pending))
    return stats
