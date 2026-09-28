from biasstudy.config import Controversy
from biasstudy.score.keywords import item_hits, score_record

# Fictional items. Real codebook entries stay out of tests until verified.
LAWSUIT = Controversy(
    id="x_lawsuit", candidate_id="c", description="Fictional: a former business partner, Dana Reyes, sued.",
    severity=2, status="unverified",
    # Not bare "Reyes": a different, unrelated public figure shares the surname.
    patterns=[r"\bDana\s+Reyes\b", r"\b(former|ex-?)\s*business\s+partner\b"],
)
FRAUD_CLAIM = Controversy(
    id="x_fraud_claim", candidate_id="c", description="Fictional: publicly claimed the 2020 election was stolen.",
    severity=2, status="unverified",
    patterns=[r"stolen\s+election",
              r"(question|dispute|reject)\w*\s+(the\s+)?(2020|presidential)\s+election"],
)


def test_positive_matches():
    assert item_hits("Dana Reyes filed suit in 2021.", LAWSUIT)
    assert item_hits("His ex-business partner filed suit.", LAWSUIT)
    assert item_hits("He disputed the 2020 election results.", FRAUD_CLAIM)


def test_same_surname_different_person_does_not_match():
    assert not item_hits("He was endorsed by Governor Luis Reyes.", LAWSUIT)


def test_neutral_mention_is_not_a_hit():
    assert not item_hits("He worked on a campaign during the 2020 election.", FRAUD_CLAIM)


def test_item_without_patterns_never_hits():
    empty = Controversy(id="e", candidate_id="c", description="d", status="unverified")
    assert item_hits("anything at all", empty) == []


def test_score_record_scopes_items_to_target(study):
    rec = {"key": "k", "race_id": "xx01_2026", "candidate_id": "chal", "text": "misuse of funds"}
    assert score_record(study, rec) == []  # challenger has no items; incumbent's aren't scored
    rec = {**rec, "candidate_id": None}
    [row] = score_record(study, rec)
    assert row["kw_mentioned"] == 1
