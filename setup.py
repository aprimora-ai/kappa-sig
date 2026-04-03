from setuptools import setup, find_packages

setup(
    name="kappa-fin",
    version="0.1.0",
    author="David Ohio",
    author_email="odavidohio@gmail.com",
    description="Topological Early Warning System for Financial Market Crises",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/odavidohio/kappa-fin",
    license="CC BY 4.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24",
        "pandas>=1.5",
        "scipy>=1.10",
        "gudhi>=3.7",
        "networkx>=3.0",
        "yfinance>=0.2",
        "matplotlib>=3.6",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Office/Business :: Financial",
    ],
)
