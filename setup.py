import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "PIA",
    version = "1.0.1",
    author = "Micha Johannes Birklbauer",
    author_email = "micha.birklbauer@gmail.com",
    description = "PIA - Protein Interaction Analyzer. Extract protein-ligand interactions and their frequencies to score and predict the activity of a complex.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/michabirklbauer/pia",
    license = "MIT",
    packages = setuptools.find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
    ]
)
