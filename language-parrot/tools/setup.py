"""
Setup script for Tuti Parrot
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent.parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding='utf-8')

setup(
    name="tuti-parrot",
    version="0.1.0",
    description="Learn languages through parroting - repetition, mimicry, and pattern recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Language Parrot Project",
    python_requires=">=3.8",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'tuti=tuti_parrot.cli:main',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Topic :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
