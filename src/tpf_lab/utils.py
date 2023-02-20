"""This module contains functionality to make the progress bars work in jupyter
notebooks.
"""

from IPython.core.display import HTML, display


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
