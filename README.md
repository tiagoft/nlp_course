# My course on NLP

A course on modern NLP

If you are going to *develop* from this repository, go to the [development guide](README_DEV.md).

## Installing My course on NLP:

Remember to follow these instructions from within your preferred virtual environment:

    conda create -n nlp_course python=3.12
    conda activate nlp_course

The first way  is to clone the repository and do a local installation:

    git clone https://github.com/tiagoft/nlp_course.git
    cd nlp_course
    pip install .

The second way is to install directly:

    pip install git+https://github.com/tiagoft/nlp_course.git

To uninstall, use:

    pip uninstall nlp_course

## Usage

To find all implemented commands, run:

    nlp_course-cli --help
