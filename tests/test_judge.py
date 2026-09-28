import json
from types import SimpleNamespace

import pytest

from biasstudy.config import JudgeSpec
from biasstudy.score.judge import Judge, JudgeOutput, build_user_message, validate


class FakeChat:
    def __init__(self, replies):
        self.replies = list(replies)

    async def create(self, **kwargs):
        content = self.replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def fake_client(replies):
    return SimpleNamespace(chat=SimpleNamespace(completions=FakeChat(replies)))


GOOD = json.dumps({"items": [{"item_id": "inc_scandal", "mentioned": 1, "framing": 2, "exculpatory": 0,
                              "factual_error": 0, "evidence": "misuse of funds"}],
                   "unlisted_negative_claims": [], "refusal": 0})
INCONSISTENT = GOOD.replace('"framing": 2', '"framing": 0')

SPEC = JudgeSpec(key="j", base_url="http://x", model_id="m", api_key_env="NONE")
REC = {"key": "b_1", "race_id": "xx01_2026", "candidate_id": None, "prompt_id": "race_1",
       "text": "She faced claims of misuse of funds."}


def test_user_message_contains_items_not_model(study):
    msg = build_user_message(study, REC)
    assert "inc_scandal" in msg and "misuse of funds" in msg


def test_validate_rejects_inconsistent_and_missing():
    with pytest.raises(ValueError):
        validate(JudgeOutput.model_validate_json(INCONSISTENT), {"inc_scandal"})
    with pytest.raises(ValueError):
        validate(JudgeOutput.model_validate_json(GOOD), {"inc_scandal", "other"})


async def test_judge_retries_bad_output(study, monkeypatch):
    import biasstudy.score.judge as judge_mod
    from tenacity import wait_none

    monkeypatch.setattr(judge_mod, "wait_random_exponential", lambda **_: wait_none())
    j = Judge(SPEC, client=fake_client(["not json", INCONSISTENT, GOOD]))
    out = await j.grade(study, REC)
    assert out.items[0].framing == 2
