from biasstudy.collect import build_jobs, run_jobs
from biasstudy.providers.base import Response
from biasstudy.store import Store


class FakeProvider:
    def __init__(self, spec, fail_times: int = 0):
        self.spec = spec
        self.calls = 0
        self.fail_times = fail_times

    async def complete(self, req):
        self.calls += 1
        if self.calls <= self.fail_times:
            err = RuntimeError("rate limited")
            err.status_code = 429
            raise err
        return Response(text=f"answer to: {req.prompt}", model_id_reported=self.spec.model_id)


def test_matrix_respects_capabilities(study):
    jobs = build_jobs(study, reps=2, arms=("nosearch", "search"), temp_modes=("default", "zero"))
    assert not any(j.model_key == "m_c" and j.arm == "search" for j in jobs)
    assert not any(j.model_key == "m_b" and j.temp_mode == "zero" for j in jobs)
    assert not any(j.model_key == "off" for j in jobs)             # disabled in config
    assert not any(j.prompt_id == "profile_2" for j in jobs)       # disabled prompt
    # chat models get both styles: (1 race + 2 profile) chat + (1 race + 2 profile) search, x2 reps
    assert sum(j.model_key == "m_a" and j.arm == "nosearch" and j.temp_mode == "default" for j in jobs) == 12
    # Google AI Overview: search arm and search-style queries only
    aio = [j for j in jobs if j.model_key == "aio"]
    assert {j.arm for j in aio} == {"search"} and {j.prompt_id for j in aio} == {"s_race", "s_profile"}
    assert len({j.key for j in jobs}) == len(jobs)


def test_disabled_model_runs_when_named(study):
    assert any(j.model_key == "off" for j in build_jobs(study, models=["off"], reps=1))


def test_search_query_rendering(study):
    jobs = build_jobs(study, models=["aio"], reps=1)
    texts = {j.prompt_text for j in jobs if j.prompt_id == "s_profile"}
    assert texts == {"Alice Incumbent Testland congresswoman", "Bob Challenger"}  # empty hint stripped


def test_prompt_order_is_deterministic(study):
    a = build_jobs(study, reps=3)
    b = build_jobs(study, reps=3)
    assert [j.prompt_text for j in a] == [j.prompt_text for j in b]
    race_jobs = [j for j in a if j.candidate_id is None]
    assert len({j.candidate_order for j in race_jobs}) == 2  # both orders occur


async def test_resumable_no_duplicates(study, tmp_path):
    jobs = build_jobs(study, models=["m_a", "m_b", "m_c"], reps=2, arms=("nosearch",))
    providers = {m.key: FakeProvider(m) for m in study.models}
    store, errors = Store(tmp_path / "raw.jsonl"), Store(tmp_path / "err.jsonl")

    first = await run_jobs(jobs[:5], providers, store, errors)
    assert first["ok"] == 5
    # Simulate a crash mid-write: truncated trailing line.
    with store.path.open("a") as f:
        f.write('{"key": "partial", "tex')
    second = await run_jobs(jobs, providers, store, errors)
    assert second == {"ok": len(jobs) - 5, "error": 0, "skipped": 5}
    keys = [r["key"] for r in store]
    assert sorted(keys) == sorted(j.key for j in jobs)


async def test_retries_rate_limits(study, tmp_path, monkeypatch):
    import biasstudy.collect as collect
    from tenacity import wait_none

    monkeypatch.setattr(collect, "wait_random_exponential", lambda **_: wait_none())
    jobs = build_jobs(study, models=["m_a"], reps=1, arms=("nosearch",))[:1]
    spec = study.model("m_a")
    stats = await run_jobs(jobs, {"m_a": FakeProvider(spec, fail_times=2)},
                           Store(tmp_path / "r.jsonl"), Store(tmp_path / "e.jsonl"))
    assert stats["ok"] == 1
