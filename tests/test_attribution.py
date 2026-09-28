from biasstudy.score.attribution import attribute


def test_sections_pronouns_and_both(study):
    race = study.race("xx01_2026")
    text = """## Overview
This race is competitive. Both parties are spending heavily.

## Alice Incumbent
Alice Incumbent was first elected in 2022. She sits on the budget committee.
- Supports tax cuts for small business owners.

**Bob Challenger**
Challenger is a former teacher. He emphasizes education funding.

Alice Incumbent and Bob Challenger will debate in October.
"""
    att = attribute(text, race)
    assert att.words["inc"] == 7 + 6 + 7   # named sentence, pronoun sentence, bullet in section
    assert att.words["chal"] == 5 + 4      # bold-heading section
    assert att.words["both"] == 9
    assert att.words["neutral"] == 4 + 5   # the untitled "Overview" section
    assert att.share("inc", "chal") == 20 / 29


def test_empty_text(study):
    att = attribute("", study.race("xx01_2026"))
    assert att.total == 0
    assert att.share("inc", "chal") is None
