from setuptools import find_packages,setup
from typing import List

HYPHEM_E_DOT = '-e .'

def get_requirments(file_path:str)->List[str]:
    """
    This function will return the list of requirements
    """

    requirements=[]

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n"," ") for req in requirements]

        if HYPHEM_E_DOT in requirements:
            requirements.remove(HYPHEM_E_DOT)
    
    return requirements

setup(
    name = 'well-spacing-analyzer',
    version = '0.0.1',
    author = 'apoorva',
    author_email = 'apoorvasaxena207@gmail.com',
    packages = find_packages(),
    install_requires = get_requirments('requirements.txt')
)
