import os
import sys
from setuptools import setup, find_packages

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))
    
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='mj_allegro_envs',
    version='1.0.3',
    packages=find_packages(),
    include_package_data = True,
    description='environments with Allegro hands for DAPG & manipulation tasks',
    long_description=read('README.md'),
    url='https://github.com/ssilwal',
    author='S',
    install_requires=[
        'click', 'gym==0.13', 'mujoco-py', 'termcolor',
    ],
)
