FROM mambaorg/micromamba

RUN micromamba install -n base -y -c conda-forge cudatoolkit==11.8 pip python==3.11 pip freetype-py

RUN micromamba clean --all --yes

#if you need to run pip install in the same environment, uncomment the following lines

ARG MAMBA_DOCKERFILE_ACTIVATE=1

RUN pip install methylbert
