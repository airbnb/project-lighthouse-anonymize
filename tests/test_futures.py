"""
Tests for futures
"""

from concurrent.futures.process import ProcessPoolExecutor

import pytest

from project_lighthouse_anonymize.futures import InProcessResult, make_future


def sample_func_for_executor(x):
    """Module-level function that can be pickled by ProcessPoolExecutor"""
    return x * 2


def sample_func_with_kwargs(x, y=None):
    """Module-level function with kwargs"""
    return x + (y or 0)


def failing_func():
    """Module-level function that raises exception"""
    raise ValueError("test error")


class TestInProcessResult:
    """Test InProcessResult class"""

    def test_init(self):
        """Test InProcessResult initialization"""

        def sample_func(x, y, z=None):
            return x + y + (z or 0)

        result = InProcessResult(sample_func, (1, 2), {"z": 3})
        assert result.func == sample_func
        assert result.args == (1, 2)
        assert result.kwargs == {"z": 3}

    def test_result_with_args_kwargs(self):
        """Test result() method with args and kwargs"""

        def sample_func(x, y, z=None):
            return x + y + (z or 0)

        result = InProcessResult(sample_func, (1, 2), {"z": 3})
        assert result.result() == 6

    def test_result_no_args_kwargs(self):
        """Test result() method with no arguments"""

        def sample_func():
            return "no args"

        result = InProcessResult(sample_func, (), {})
        assert result.result() == "no args"

    def test_cancel(self):
        """Test cancel() method"""

        def sample_func():
            return "test"

        result = InProcessResult(sample_func, (), {})
        assert result.cancel() is True


class TestMakeFuture:
    """Test make_future function"""

    def test_make_future_with_executor(self):
        """Test make_future with ProcessPoolExecutor"""
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = make_future(executor, sample_func_for_executor, 5)
            # Should return actual Future object
            assert hasattr(future, "result")
            assert hasattr(future, "cancel")
            assert future.result() == 10

    def test_make_future_without_executor(self):
        """Test make_future with None executor"""
        future = make_future(None, sample_func_with_kwargs, 5, y=3)
        # Should return InProcessResult
        assert isinstance(future, InProcessResult)
        assert future.result() == 8

    def test_make_future_exception_handling(self):
        """Test make_future handles function exceptions"""
        future = make_future(None, failing_func)
        with pytest.raises(ValueError, match="test error"):
            future.result()
