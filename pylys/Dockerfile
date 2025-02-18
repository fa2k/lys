FROM continuumio/miniconda3
COPY environment.yml /
RUN conda env create -f /environment.yml
COPY output_adapters.py main.py /
CMD ["/opt/conda/envs/pylys/bin/python", "/main.py"]
