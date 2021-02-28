from setuptools import setup, find_packages
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

version = {}
with open("piper/version.py") as fp:
    exec(fp.read(), version)

setup(
    name='dpiper',
    version=version["__version__"],
    author='Mike Tarpey',
    author_email='miketarpey@gmx.net',
    url='https://github.com/miketarpey/piper',
    packages=find_packages(),
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.20.0",
        "openpyxl>=3.0.6",
        "seaborn>=0.11.1",
        "xlsxwriter>=1.3.2",
        "cx_oracle",
        "psycopg2",
        "pypyodbc"],
    tests_require=['pytest'],
    include_package_data=True,
    license='BSD',
    description='A Python module for maintaining pipeline syntax of Pandas statements.',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)

