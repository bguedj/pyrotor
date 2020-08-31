import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(name='pyrotor',
      version='1.0.0',
      description='Trajectory optimization package based on data',
      long_description=README,
      packages=['pyrotor'],
      package_data={
          "pyrotor": ["toy_dataset/*.csv"]
      },
      author_email="arthur.talpaert@isen.yncrea.fr",
      url="https://github.com/bguedj/pyrotor",
      license="MIT",
      install_requires=["cvxopt>=1.2.4",
                        "numpy>=1.19.1",
                        "pandas>=0.25.3",
                        "scipy>=1.5.2",
                        "mock>=3.0.5",
                        "pytest>=5.3.1",
                        "pickleshare>=0.7.5",
                        "scikit-learn>=0.23"])
