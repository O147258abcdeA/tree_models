from setuptools import setup, find_packages

setup(
    name="tree_models",
    version="0.1.0",
    description="Combine multiple tree-based ML models into a single combo estimator",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.8",
    install_requires=[
        "scikit-learn>=1.0",
        "numpy>=1.21",
    ],
)
