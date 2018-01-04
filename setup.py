from setuptools import setup, find_packages

# Convert the Markdown to Restructuredtext for PyPI
try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
    long_description = long_description.replace("\r", "")
except (IOError, ImportError):
    print('error')
    long_description = ''

setup(
    name = "odenlls",
    version = "0.1.0",

    description = "Non-linear least squares fitting of chemical " + \
            "kinetics data using ODE simulations",

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

    keywords = "non-linear fitting chemical kinetics ordinary " + \
            "differential equations ode",

    packages = find_packages(),
    install_requires = [
        'numpy>=1.13.3',
        'scipy>=1.0.0',
        'pandas>=0.21.1',
        'matplotlib>=2.1.1',
    ],

)

