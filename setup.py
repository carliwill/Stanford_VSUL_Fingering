"""
co2_fingers — setup.py
"""

from setuptools import setup, find_packages

setup(
    name="co2_fingers",
    version="0.1.0",
    description="CO₂ convective fingering analysis for FluidFlower experiments",
    author="Carlin Will",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "opencv-python",
        "matplotlib",
    ],
)
