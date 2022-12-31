import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(name='pyrotor',
      version='1.0.4',
      description='Trajectory optimization package based on data',
      long_description=README,
      long_description_content_type='text/markdown',
      packages=['pyrotor'],
      package_data={
          "pyrotor": ["toy_dataset/example_1/*.csv",
                      "toy_dataset/example_2/*.csv"]
      },
      author_email="arthur.talpaert@isen.yncrea.fr",
      url="https://github.com/bguedj/pyrotor",
      license="MIT",
      install_requires=["cvxopt>=1.2.7",
                        "numpy>=1.24.1",
                        "pandas>=1.5.2",
                        "scipy>=1.5.4",
                        "mock",
                        "pytest",
                        "pickleshare>=0.7.5",
                        "scikit-learn>=1.2.0",
                        "sphinx",
                        "sphinx-rtd-theme"])
