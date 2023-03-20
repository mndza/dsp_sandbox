from setuptools import setup
from setuptools_scm.git import parse as parse_git

def doc_version():
    git = parse_git(".")
    if git.exact:
        return git.format_with("v{tag}")
    else:
        return "latest"

setup()