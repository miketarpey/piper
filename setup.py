import sys
import subprocess
from warnings import warn

try:
    from setuptools import setup, Command
except ImportError:
    from distutils.core import setup, Command

if sys.version_info < (3, 7, 0):
    warn("The minimum Python version supported by piper is 3.7.")
    exit()

setup(
    name='piper',
    version='0.0.1',
    author='Mike Tarpey',
    author_email='miketarpey@gmx.net',
    url='https://github.com/miketarpey/piper',
    packages=['piper'],
    license='BSD',
    description='A Python module for maintaining pipeline syntax of Pandas statements.',
    long_description=open('README.rst').read(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)

