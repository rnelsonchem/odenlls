from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

setup(
    name = "ODEnlls",
    version = "0.1",

    description = "Non-linear least squares fitting using ODEs ' + \
            'for chemical kinetics.",
    long_description = long_description,
    url = "https://github.com/rnelsonchem/ODEnlls",

    author = "Ryan Nelson",
    author_email = "rnelsonchem@gmail.com",

    license = "MIT",
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],

    keywords = "fitting chemical kinetics ordinary differential " + \
            "equations ode",

    packages = find_packages(),
    install_requires = [
        'numpy>=1.13.3',
        'scipy>=1.0.0',
        'pandas>=0.21.1',
        'matplotlib>=2.1.1',
    ],

)

