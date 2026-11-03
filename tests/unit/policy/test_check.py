"""check_discount_policy: sprawdzenie polityki tylko do odczytu nad `required_approvers`
(port 4 testów polityki z kroku 2 MP4; `config` → `settings.policy`; kontrakt F4, 6.1)."""

from __future__ import annotations


def check(settings, **kwargs):
    from kovalo.policy.check import check_discount_policy

    return check_discount_policy(settings.policy, **kwargs)


def test_policy_check_reuses_the_rules_engine(settings):
    result = check(settings, discount_pct=22, margin_after_pct=7, strategic_account=True)
    assert result.status == "ok"
    assert result.approvers == ["Dyrektor handlowy", "CFO"]
    assert result.version == "course-v1" and result.margin_checked is True
    assert result.partial is False
    assert result.reasons
    assert "niczego nie zatwierdza" in result.note


def test_an_unknown_margin_is_a_partial_check(settings):
    result = check(settings, discount_pct=14)
    assert result.approvers == ["Dyrektor handlowy"]
    assert result.margin_checked is False and result.partial is True


def test_volume_rebates_are_out_of_scope(settings):
    result = check(settings, discount_pct=4, contractual_volume_rebate=True)
    assert result.status == "out_of_scope" and result.approvers == []
    assert "warunków kredytowych i handlowych" in result.message


def test_invalid_policy_input(settings):
    result = check(settings, discount_pct=140)
    assert result.status == "invalid_request" and result.approvers == []
    assert "discount" in result.message


def test_matches_the_engine_and_never_approves(settings):
    """Te same role i uzasadnienia co `required_approvers`; nota mówi, że niczego nie zatwierdza."""
    from kovalo.policy import DiscountRequest, required_approvers

    decision = required_approvers(
        DiscountRequest(discount_pct=22, margin_after_pct=7, strategic_account=True),
        settings.policy,
    )
    result = check(settings, discount_pct=22, margin_after_pct=7, strategic_account=True)
    assert result.approvers == decision.approvers and result.reasons == decision.reasons
    assert "wnioski o akceptację przechodzą przez workflow" in result.note
