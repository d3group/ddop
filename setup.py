from setuptools import setup, find_packages
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

with open('README.md') as f:
    README = f.read()

extensions = [
    Extension("criterion",
              ['ddop/utils/criterion.pyx'],
              include_dirs=[numpy.get_include()]),
]

setup(
    name='ddop',
    version='v0.2.1',
    url='',
    license='MIT',
    author='Andreas Philippi',
    author_email='',
    description='Package for data-driven operations management',
    long_description=README,
    include_package_data=True,
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=['scikit-learn>=0.23.0', 'pandas', 'PuLP==2.0',
                      'tensorflow==2.1.0', 'Keras==2.3.1',
                      'numpy==1.18.2', 'scipy==1.4.1', 'Cython'],
    ext_modules=cythonize(extensions)
)
