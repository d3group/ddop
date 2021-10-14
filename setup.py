from setuptools import setup, find_packages

with open('README.rst') as f:
    README = f.read()

setup(
    name='ddop',
    version='v0.7.5',
    url='https://andreasphilippi.github.io/ddop/',
    license='MIT',
    author='Andreas Philippi',
    author_email='andreas.philippi@uni-wuerzburg.de',
    description='Package for data-driven operations management',
    #long_description=README,
    include_package_data=True,
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=['numpy>=1.18.2', 'scipy>=1.4.1', 'scikit-learn==0.23.0', 'pandas', 'pulp==2.0',
                      'tensorflow>=2.4.1', 'statsmodels>=0.11.1', 'mpmath']
)
