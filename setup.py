from setuptools import setup, find_packages

with open('README.md') as f:
    README = f.read()

setup(
    name='ddop',
    version='v0.0.1',
    packages=['ddop'],
    url='',
    license='MIT',
    author='Andreas Philippi',
    author_email='',
    description='Package for data-driven operations management',
    long_description=README,
    packages = find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=['sklearn>=0.0','pandas']
)