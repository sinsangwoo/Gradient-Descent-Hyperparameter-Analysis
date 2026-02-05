"""Legacy test file - DEPRECATED.

This file tested a class-based API (HeatEquation1D, WaveEquation1D)
that no longer exists. We now use a functional API.

See test_heat_equation.py for current tests.
"""

import pytest


def test_deprecated_api_notice():
    """Placeholder test to prevent collection errors."""
    # This file is deprecated but kept for historical reference
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
