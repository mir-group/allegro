from setuptools import setup, find_packages

setup(
    name="mir-allegro",
    version="0.1.0",
    author="Albert Musaelian, Simon Batzner",
    description="Allegro is an open-source code for building highly scalable and accurate equivariant deep learning interatomic potentials.",
    python_requires=">=3.7",
    packages=find_packages(include=["allegro", "allegro.*"]),
    install_requires=["nequip>=0.5.3"],
    zip_safe=True,
)
