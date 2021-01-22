from setuptools import setup, find_packages

with open('README.rst') as f:
    README = f.read()

setup(
    name='ddop',
    version='v0.6.1',
    url='https://andreasphilippi.github.io/ddom-kit/',
    license='MIT',
    author='Andreas Philippi',
    author_email='',
    description='Package for data-driven operations management',
    #long_description=README,
    include_package_data=True,
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=['scikit-learn>=0.23.0', 'pandas', 'PuLP==2.0',
                      'tensorflow>=2.1.0', 'Keras>=2.3.1', 'statsmodels',
                      'numpy>=1.18.2', 'scipy>=1.4.1', 'lightgbm>=2.3.1'],
)
