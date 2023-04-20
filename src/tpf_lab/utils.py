"""This module contains functionality to make the progress bars work in jupyter
notebooks.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator, Optional, Type  # pylint: disable=unused-import

from tqdm.std import tqdm as std_tqdm
from IPython.core.display import HTML, display
from tqdm.contrib.logging import (
    _get_first_found_console_logging_handler,
    _is_console_logging_handler,
    _TqdmLoggingHandler,
)


def rm_out_padding():
    """Call this function to prevent a closed progress bar to leave a blank line.

    Depending on the used browser, this may be needed when running PorePy in a jupyter
    notebook. However, it is not guaranteed to help, as, e.g., it does not work in
    VSCode.

    Mentioned in https://github.com/jupyter-widgets/ipywidgets/issues/1845.
    """
    display(HTML("<style>div.output_subarea { padding:unset;}</style>"))


def is_notebook() -> bool:
    """Determine whether the code is running in a Jupyter notebook.

    Copied from https://stackoverflow.com/a/39662359.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


@contextmanager
def logging_redirect_tqdm(
    loggers=None,  # type: Optional[list[logging.Logger]]
    tqdm_class=std_tqdm,  # type: Type[std_tqdm]
):
    # type: (...) -> Iterator[None]
    """
    This is a slightly modified version of
    ``tqdm.contrib.logging.logging_redirect_tqdm``, as the original function does not
    inherit the logging level. There are multiple PRs waiting for acception. As soon as
    they are implemented, this function can be deleted.
    https://github.com/tqdm/tqdm/issues/1272


    Context manager redirecting console logging to `tqdm.write()`, leaving
    other logging handlers (e.g. log files) unaffected.

    Parameters
    ----------
    loggers  : list, optional
      Which handlers to redirect (default: [logging.root]).
    tqdm_class  : optional

    Example
    -------
    ```python
    import logging
    from tqdm import trange
    from tqdm.contrib.logging import logging_redirect_tqdm

    LOG = logging.getLogger(__name__)

    if __name__ == '__main__':
        logging.basicConfig(level=logging.INFO)
        with logging_redirect_tqdm():
            for i in trange(9):
                if i == 4:
                    LOG.info("console logging redirected to `tqdm.write()`")
        # logging restored
    ```
    """
    if loggers is None:
        loggers = [logging.root]
    original_handlers_list = [logger.handlers for logger in loggers]
    try:
        for logger in loggers:
            tqdm_handler = _TqdmLoggingHandler(tqdm_class)
            orig_handler = _get_first_found_console_logging_handler(logger.handlers)
            if orig_handler is not None:
                tqdm_handler.setFormatter(orig_handler.formatter)
                tqdm_handler.stream = orig_handler.stream
                # MODIFICATION:
                # Copy the level of the original handler.
                tqdm_handler.level = orig_handler.level
            logger.handlers = [
                handler
                for handler in logger.handlers
                if not _is_console_logging_handler(handler)
            ] + [tqdm_handler]
        yield
    finally:
        for logger, original_handlers in zip(loggers, original_handlers_list):
            logger.handlers = original_handlers
