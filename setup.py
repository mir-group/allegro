from setuptools import setup, find_packages

setup(
    name="nequip-allegro",
    version="0.1.0",
    author="Albert Musaelian, Simon Batzner",
    python_requires=">=3.7",
    packages=find_packages(include=["nequip_allegro", "nequip_allegro.*"]),
    install_requires=["nequip>=0.5.3"],
    zip_safe=True,
)
