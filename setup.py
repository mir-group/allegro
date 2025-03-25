from setuptools import setup, find_packages
from pathlib import Path

# see https://packaging.python.org/guides/single-sourcing-package-version/
version_dict = {}
with open(Path(__file__).parents[0] / "allegro/_version.py") as fp:
    exec(fp.read(), version_dict)
version = version_dict["__version__"]
del version_dict

setup(
    name="mir-allegro",
    version=version,
    author="Albert Musaelian, Simon Batzner",
    description="Allegro is an open-source code for building highly scalable and accurate equivariant deep learning interatomic potentials.",
    python_requires=">=3.7",
    packages=find_packages(include=["allegro", "allegro.*"]),
    install_requires=["nequip>=0.6.1,<0.7.0"],
    zip_safe=True,
    entry_points={"nequip.extension": ["init_always = allegro"]},
)
