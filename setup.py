# setup.py for unsloth - Fast LLM fine-tuning library
# Fork of unslothai/unsloth with additional features and fixes

from setuptools import setup, find_packages
import os

# Read the README for the long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Core dependencies required for basic functionality
INSTALL_REQUIRES = [
    "torch>=2.1.0",
    "transformers>=4.38.0",
    "datasets>=2.16.0",
    "accelerate>=0.26.0",
    "peft>=0.9.0",
    "trl>=0.7.10",
    "bitsandbytes>=0.42.0",
    "sentencepiece>=0.1.99",
    "protobuf>=3.20.0",
    "huggingface_hub>=0.20.0",
    "triton>=2.1.0",
    "einops>=0.7.0",
    # Removed xformers from core deps - it's optional and often causes install issues
    # "xformers>=0.0.23",
]

# Optional dependencies for extended functionality
EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
    ],
    "flash-attn": [
        "flash-attn>=2.5.0",
    ],
    "galore": [
        "galore-torch>=1.0",
    ],
    "bnb": [
        "bitsandbytes>=0.43.0",
    ],
    "xformers": [
        "xformers>=0.0.23",
    ],
}

# Combine all extras into a 'full' option
EXTRAS_REQUIRE["full"] = [
    dep
    for key, deps in EXTRAS_REQUIRE.items()
    if key != "dev"
    for dep in deps
]

setup(
    name="unsloth",
    version="2024.1.0",
    author="Unsloth AI",
    author_email="info@unsloth.ai",
    description="2-5x faster, 70% less memory LLM fine-tuning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    # Updated to point to my personal fork
    url="https://github.com/unslothai/unsloth",
    project_urls={
        "Bug Tracker": "https://github.com/unslothai/unsloth/issues",
        "Documentation": "https://github.com/unslothai/unsloth/wiki",
        "Source Code": "https://github.com/unslothai/unsloth",
    },
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    python_requires=">=3.9",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        # Added 3.12 since it's now widely used and works fine
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: POSIX :: Linux",
    ],
    entry_points={
        "console_scripts": [
            "unsloth=unsloth.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "llm",
        "fine-tuning"
    ],
)
