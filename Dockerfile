FROM nvidia/cuda:11.1.0-cudnn8-devel-ubuntu20.04

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -ay && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/profile && \
    echo "conda activate base" >> /etc/profile

WORKDIR /root/code

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda create -n crystal_generation -y python=3.8.13 && \
    conda activate crystal_generation && \
    conda install pip -y && \
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch && \
    conda install pyg -c pyg && \
    pip install ase==3.22 && \
    pip install autopep8 && \
    pip install jupyterlab && \
    pip install matminer==0.7.3 && \
    pip install matplotlib && \
    pip install nglview && \ 
    pip install ipywidgets && \
    pip install pylint && \
    pip install pymatgen==2022.0.8 && \
    pip install seaborn && \
    pip install tqdm && \
    pip install torch-geometric==1.7.2 && \
    pip install higher==0.2.1 && \
    pip install pip install hydra-core==1.1.0 && \
    pip install hydra-core==1.1.0 && \
    pip install hydra-core==1.1.0 && \
    pip install pytest && \
    pip install pip install pytest && \
    pip install smact==2.2.1 && \
    pip install streamlit==0.79.0 && \
    pip install torchdiffeq && \
    pip install wandb==0.10.33 && \