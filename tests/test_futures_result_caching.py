"""
Tests for InProcessResult Future-interface conformance.

concurrent.futures.Future computes its result once and caches it; the
serial-execution stand-in must do the same, or repeated result() calls
re-execute the function against the current (possibly mutated) state of
shared arguments, diverging from parallel-mode behavior.
"""

import pytest

from project_lighthouse_anonymize.futures import InProcessResult, make_future


class TestInProcessResultCaching:
    """Tests that InProcessResult executes its function exactly once"""

    def test_result_executes_once(self):
        """Repeated result() calls must not re-execute the function"""
        call_count = []

        def func(value):
            call_count.append(1)
            return value * 2

        in_process_result = InProcessResult(func, (21,), {})
        first = in_process_result.result()
        second = in_process_result.result()
        assert first == 42
        assert second == 42
        assert len(call_count) == 1, f"function executed {len(call_count)} times"

    def test_make_future_serial_path_caches(self):
        """make_future with no executor returns a caching result object"""
        state = {"value": 1}

        def read_state():
            return state["value"]

        future = make_future(None, read_state)
        assert future.result() == 1
        state["value"] = 2
        assert future.result() == 1, "result() re-executed against mutated state"


class TestInProcessResultExceptionCaching:
    """Tests that InProcessResult caches and re-raises exceptions"""

    def test_exception_cached_and_reraised(self):
        """A raised exception is cached and re-raised, matching Future

        concurrent.futures.Future stores the exception and re-raises it on
        every result() call; re-executing the function could flip from failure
        to success against mutated state.
        """
        call_count = []

        def failing_func():
            call_count.append(1)
            raise ValueError("transient failure")

        in_process_result = InProcessResult(failing_func, (), {})
        with pytest.raises(ValueError, match="transient failure"):
            in_process_result.result()
        with pytest.raises(ValueError, match="transient failure"):
            in_process_result.result()
        assert len(call_count) == 1, f"function executed {len(call_count)} times"
