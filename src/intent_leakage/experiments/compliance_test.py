"""Compatibility re-export of the canonical Section 7.3 compliance API.

New code should import :mod:`intent_leakage.compliance` directly. Keeping this
module as a pure re-export avoids two divergent definitions of the seven-step
test while preserving imports from early repository drafts.
"""

from ..compliance import (
    CheckStatus,
    ComplianceCheck,
    ComplianceEvidence,
    ComplianceReport,
    ComplianceThresholds,
    evaluate_compliance,
)

__all__ = [
    "CheckStatus",
    "ComplianceCheck",
    "ComplianceEvidence",
    "ComplianceReport",
    "ComplianceThresholds",
    "evaluate_compliance",
]
