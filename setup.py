#!/usr/bin/env python3
from setuptools import setup, find_packages
import os

# Read requirements.txt
requirements = []
requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
if os.path.exists(requirements_path):
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f.readlines()
                      if line.strip() and not line.startswith('#')]

setup(
    name="challenge_solution_projet_ECE",
    version="1.0.0",
    description="Welding Quality Detection AI Component",
    author="Challenge Participant",
    packages=find_packages(),
    package_data={
        'challenge_solution': ['*.pth', '*.pkl', '*.json', '*.txt'],
    },
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.8",
    zip_safe=False,
)