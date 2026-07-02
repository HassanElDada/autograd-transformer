# A minimal 'smoke test' - it doesn't test our real code yet.
# Its only job is to prove the pytest pipeline works end to end:
# that pytest can discover this file, run the function, and report a result.

def test_smoke():
    # 'assert' checks that a condition is True.
    # If it's True, the test passes. If False, pytest reports a failure.
    assert 1 + 1 == 2
