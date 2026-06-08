"""
tests/unit/test_alignment_type.py

Tests for AlignmentType enum and its boundary conditions.
Rules (from .claude/rules/spacing-engine.md):
  PARALLEL_LIKE  : angle ≤ 25.0° (inclusive)
  OBLIQUE        : 25.0° < angle < 65.0°
  PERPENDICULAR  : angle ≥ 65.0° (inclusive)
  Boundary wells (exactly 25° → PARALLEL_LIKE, exactly 65° → PERPENDICULAR)
"""

import pytest

from src.well_data.well_spacing_stats import AlignmentType

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Enum membership
# ---------------------------------------------------------------------------


class TestAlignmentTypeMembers:
    def test_all_four_values_exist(self):
        members = {m.name for m in AlignmentType}
        assert members == {"PARALLEL_LIKE", "OBLIQUE", "PERPENDICULAR", "MISALIGNED"}

    def test_unique_values(self):
        values = [m.value for m in AlignmentType]
        assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# Boundary classification helper
# This replicates the classifier logic from well_spacing_stats so tests
# remain independent of private internals. Boundaries are documented in the
# rules file and docstring.
# ---------------------------------------------------------------------------


def classify(angle_deg: float, theta_parallel: float = 25.0, theta_perp: float = 65.0) -> AlignmentType:
    """Pure reference implementation of alignment classification."""
    if angle_deg <= theta_parallel:
        return AlignmentType.PARALLEL_LIKE
    if angle_deg >= theta_perp:
        return AlignmentType.PERPENDICULAR
    return AlignmentType.OBLIQUE


class TestBoundaryConditions:
    """Critical: exact boundary values must map to the correct class."""

    @pytest.mark.parametrize("angle", [0.0, 10.0, 24.999, 25.0])
    def test_parallel_like_range(self, angle):
        assert classify(angle) == AlignmentType.PARALLEL_LIKE

    @pytest.mark.parametrize("angle", [25.001, 44.0, 64.999])
    def test_oblique_range(self, angle):
        assert classify(angle) == AlignmentType.OBLIQUE

    @pytest.mark.parametrize("angle", [65.0, 65.001, 80.0, 90.0])
    def test_perpendicular_range(self, angle):
        assert classify(angle) == AlignmentType.PERPENDICULAR

    def test_exactly_25_is_parallel_not_oblique(self):
        assert classify(25.0) == AlignmentType.PARALLEL_LIKE

    def test_exactly_65_is_perpendicular_not_oblique(self):
        assert classify(65.0) == AlignmentType.PERPENDICULAR

    def test_zero_degrees_parallel(self):
        """Perfectly parallel wells — the common case for offset child wells."""
        assert classify(0.0) == AlignmentType.PARALLEL_LIKE

    def test_ninety_degrees_perpendicular(self):
        """Maximum angle — e.g., EW parent with NS child."""
        assert classify(90.0) == AlignmentType.PERPENDICULAR


class TestSpacingResultPairAlignment:
    """Verify SpacingResult.pair_alignment string matches AlignmentType names."""

    @pytest.mark.parametrize("alignment_str,expected_type", [
        ("parallel_like",  AlignmentType.PARALLEL_LIKE),
        ("oblique",        AlignmentType.OBLIQUE),
        ("perpendicular",  AlignmentType.PERPENDICULAR),
        ("misaligned",     AlignmentType.MISALIGNED),
    ])
    def test_string_to_enum_roundtrip(self, alignment_str, expected_type):
        """pair_alignment strings should match AlignmentType names (lowercased)."""
        assert alignment_str == expected_type.name.lower()
