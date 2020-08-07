import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(name='pyrotor',
      version='1.0.0',
      description='Trajectory optimization package based on data',
      long_description=README,
      packages=['pyrotor'],
      author_email="arthur.talpaert@isen.yncrea.fr",
      url="https://github.com/bguedj/pyrotor",
      license="MIT",
      install_requires=["cvxopt>=1.2.4",
                        "numpy>=1.17.4",
                        "pandas>=0.25.3",
                        "scipy>=1.3.1"])
