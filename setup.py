from setuptools import find_packages,setup
from typing import List

HYPHEM_E_DOT = '-e .'

def get_requirments(file_path:str)->List[str]:
    """
    This function will return the list of requirements

    Strips comments, blank lines, and the `-e .` self-reference (keeping it
    would make install_requires point the project at itself).
    """

    requirements=[]

    with open(file_path) as file_obj:
        for line in file_obj:
            req = line.split("#", 1)[0].strip()
            if not req or req == HYPHEM_E_DOT:
                continue
            requirements.append(req)

    return requirements

setup(
    name = 'well-spacing-analyzer',
    version = '0.0.1',
    author = 'apoorva',
    author_email = 'apoorvasaxena207@gmail.com',
    packages = find_packages(),
    install_requires = get_requirments('requirements.txt')
)
