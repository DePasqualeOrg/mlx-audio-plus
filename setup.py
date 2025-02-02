from setuptools import setup, find_packages

setup(
    name="mlx-tts",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        # List your package dependencies here
    ],
    author="Prince Canuma",
    author_email="princecanuma@gmail.com",
    description="MLX-TTS is a package for inference of TTS models locally on your Mac using MLX",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Blaizzy/mlx-tts",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)