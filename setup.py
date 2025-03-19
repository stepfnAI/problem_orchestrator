#!/usr/bin/env python
"""
Setup script for the Meta-Agent framework.
"""

from setuptools import setup, find_packages

setup(
    name="meta-agent",
    version="0.1.0",
    description="Meta-Agent framework for automated data science",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "meta-agent=cli:main",
            "meta-agent-run=main:main",
        ],
    },
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
    ],
    python_requires=">=3.8",
) 