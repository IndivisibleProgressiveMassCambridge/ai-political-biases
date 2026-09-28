import pytest
from pydantic import ValidationError

from biasstudy.config import Controversy, load_study


def test_real_config_loads():
    study = load_study()
    assert any(r.id == "oh07_2026" for r in study.races)
    assert study.race("oh07_2026").incumbent.id == "max_miller"
    # Index-case items are placeholders until a human verifies them.
    assert any(c.candidate_id == "max_miller" for c in study.unverified)


def test_verified_item_requires_sources_and_date():
    with pytest.raises(ValidationError):
        Controversy(id="x", candidate_id="c", description="d", severity=1,
                    status="verified", sources=["one"], patterns=["x"])


def test_real_codebook_patterns_compile():
    import re

    for item in load_study().controversies:
        for p in item.patterns:
            re.compile(p)
