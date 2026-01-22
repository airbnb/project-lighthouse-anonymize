"""
Utility functions for concurrent.futures.

This module provides utilities to work with Python's concurrent.futures module
for parallel and deferred execution.
"""

from concurrent.futures import Future
from concurrent.futures.process import ProcessPoolExecutor
from typing import Any, Callable, Optional, Union


class InProcessResult:
    """
    A simplified implementation of concurrent.futures.Future interface.

    This class provides a minimal implementation of the Future interface
    that executes the function in-process when result() is called, rather
    than executing in a separate process or thread.

    Parameters
    ----------
    func : callable
        The function to execute
    args : tuple
        Positional arguments to pass to the function
    kwargs : dict
        Keyword arguments to pass to the function
    """

    def __init__(
        self, func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def result(self) -> Any:
        """
        Execute the function and return its result.

        Returns
        -------
        Any
            The result of calling the function with the provided arguments
        """
        return self.func(*self.args, **self.kwargs)

    def cancel(self) -> bool:
        """
        Placeholder implementation of Future.cancel().

        Returns
        -------
        bool
            Always returns True as there's nothing to cancel
        """
        return True


def make_future(
    executor: Optional[ProcessPoolExecutor], func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Union[Future, InProcessResult]:
    """
    Create a Future-like object for concurrent or deferred execution.

    This function creates either a real Future (if an executor is provided)
    or a placeholder that mimics the Future interface (if no executor).
    The latter executes synchronously when its result() method is called.

    Parameters
    ----------
    executor : ProcessPoolExecutor or None
        If not None, the function will execute concurrently using this executor.
        If None, the function will execute in the current process when future.result() is called.
    func : callable
        The function to execute.
    *args : Any
        Positional arguments to pass to the function.
    **kwargs : Any
        Keyword arguments to pass to the function.

    Returns
    -------
    Union[Future, InProcessResult]
        A Future-like object that will execute the function when its result() method is called.
    """
    if executor is not None:
        return executor.submit(func, *args, **kwargs)
    return InProcessResult(func, args, kwargs)
