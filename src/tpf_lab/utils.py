from IPython.core.display import HTML, display


def rm_out_padding():
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
