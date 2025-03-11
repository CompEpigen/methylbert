FROM mambaorg/micromamba:cuda11.8.0-ubuntu20.04

RUN micromamba install -n base -y -c conda-forge pip python==3.11 pip freetype-py

RUN micromamba clean --all --yes

#if you need to run pip install in the same environment, uncomment the following lines

ARG MAMBA_DOCKERFILE_ACTIVATE=1

RUN mkdir src/
RUN mkdir src/methylbert/
COPY src/methylbert/ src/methylbert/
COPY pyproject.toml .
RUN pip install .
