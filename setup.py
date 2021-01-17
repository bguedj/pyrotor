import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(name='pyrotor',
      version='1.0.3',
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
      install_requires=["cvxopt==1.2.4",
                        "numpy==1.19.1",
                        "pandas==0.25.3",
                        "scipy==1.5.2",
                        "mock==3.0.5",
                        "pytest==5.3.1",
                        "pickleshare==0.7.5",
                        "scikit-learn==0.23.2",
                        "sphinx==2.3.1",
                        "recommonmark==0.6.0",
                        "sphinx-rtd-theme==0.5.0"])
