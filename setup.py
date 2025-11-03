from setuptools import setup, find_packages

setup(
    name="streamax",  # name on PyPI / pip
    version="0.2.0",
    author="David",
    author_email="dc824@cam.ac.uk",
    description="A JAX-accelerated stream generator (StreaMAX)",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/David-Chemaly/StreaMAX",
    packages=find_packages(exclude=["notebooks*"]),
    install_requires=[
        "jax>=0.4.0",
        "astropy>=5.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # or your license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    include_package_data=True,
)
