FROM mambaorg/micromamba

RUN micromamba install -n base -y -c conda-forge cudatoolkit==11.1.1 pip python==3.7
RUN micromamba clean --all --yes

#if you need to run pip install in the same environment, uncomment the following lines

ARG MAMBA_DOCKERFILE_ACTIVATE=1

RUN pip install methylbert==0.0.2rc0
RUN pip install urllib3==1.26.6
