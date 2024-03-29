# Dockerfile with the PIA package installed and PIAScript
# author: Micha Birklbauer
# version: 1.0.1

FROM ubuntu:20.04

LABEL maintainer="micha.birklbauer@gmail.com"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    libopenbabel-dev \
    libopenbabel6 \
    openbabel \
    python3-openbabel \
    pymol \
    python3-pymol \
    python3-distutils \
    python3-lxml \
    python3-rdkit \
    python3-pip

RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install scikit-learn
RUN pip3 install plip==2.2.2 --no-deps
RUN pip3 install biopandas
RUN pip3 install matplotlib
RUN pip3 install streamlit
RUN pip3 install jupyterlab

RUN git clone https://github.com/michabirklbauer/pia.git
WORKDIR pia
RUN python3 setup.py install
WORKDIR /

RUN mkdir exchange
RUN mkdir PIA

COPY PIAScript.py PIA

CMD  ["/bin/bash"]
