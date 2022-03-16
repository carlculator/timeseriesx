#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

setup_requirements = []

test_requirements = ['pytest>=3', ]

requirements = [
    'pandas>=0.25',
    'pint-pandas>=0.2',
    'pytz>=2020.5',
    'python-dateutil>=2.8.1',
]

setup(
    author="Alexander Schulz",
    author_email='info@alexander-schulz.eu',
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description="Manage time series data with explicit frequency and unit.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='timeseriesx',
    name='timeseriesx',
    packages=find_packages(include=['timeseriesx', 'timeseriesx.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/carlculator/timeseriesx',
    version='0.1.12',
    zip_safe=False,
)
