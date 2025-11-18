from setuptools import setup, find_packages

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]


setup(name='urban_watch',
      version="0.1.0",
      packages=find_packages(),   # détecte automatiquement urban_watch et ses sous-packages
      install_requires=requirements,
      description="Combining the power of Sentinel-2 satellites and machine learning to automate urbanisation analysis")
