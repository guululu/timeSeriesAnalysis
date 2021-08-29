import os


def get_project_dir() -> str:
    """
    Get the full path to the repository
    """

    current_dir = os.getcwd()

    while True:
        if os.path.basename(current_dir) != 'timeSeriesAnalysis':
            current_dir = os.path.dirname(current_dir)
        else:
            return current_dir


def get_file(relative_path: str) -> str:
    """
    Given the relative path to the repository, return the full path

    """
    return os.path.join(get_project_dir(), relative_path)