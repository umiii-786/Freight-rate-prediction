"""Custom exception + a decorator that logs the traceback once and re-raises."""
import functools
import sys


class PipelineException(Exception):
    """Wraps any error and adds the file name + line number where it happened."""

    def __init__(self, error: Exception | str):
        _, _, tb = sys.exc_info()
        self.file, self.line = "<unknown>", 0
        while tb is not None:                # walk to the innermost frame
            self.file = tb.tb_frame.f_code.co_filename
            self.line = tb.tb_lineno
            tb = tb.tb_next
        self.original = error
        super().__init__(f"[{self.file}:{self.line}] {type(error).__name__}: {error}")


def handle_errors(logger, action: str):
    """Decorator: log the failure (with traceback) once and raise PipelineException.

        @handle_errors(logger, "reading the CSV")
        def read(...): ...
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except PipelineException:        # already logged deeper in the stack
                raise
            except Exception as e:
                logger.exception("Failed while %s", action)
                raise PipelineException(e) from e

        return wrapper

    return decorator
