"""
Setup configuration for the Federated Learning System.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="federated-learning-system",
    version="0.1.0",
    author="Federated Learning Team",
    description="Privacy-Preserving Federated Learning for Image Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Prashant-ambati/Federated-Learning-for-Privacy-Preserving-Image-Classification",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.2.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.1.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "aws": [
            "boto3>=1.26.0",
            "awscli>=1.27.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fl-coordinator=src.coordinator.main:main",
            "fl-client=src.client.main:main",
        ],
    },
)