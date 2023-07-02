# Explaining and approximating a graph convolutional network trained on the IMDB-BINARY dataset
The GCN trained on IMDB-BINARY achieves an accuracy of up to 0.76. Using a custom graph class ```graph.py``` to access implicit properties in graphs from the IMDB-BINARY dataset enables an explanation and approximation of the GCN model as well as a direct classification accuracy of 0.71 on the dataset.

## Setup
### Requirements
This project was run with Python 3.11.3 under Windows 10 (64-bit). Dependencies and exact package versions are listed in requirements.txt.
### Installation
1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Create conda environment with:
   ```conda create --name imdb-gcn python=3.11.3```
3. Activate conda environment: ```conda activate imdb-gcn```
4. Install [PyTorch](https://pytorch.org/) in activated conda environment. We used the following command to install PyTorch version 2.0.1:
   ```
   conda install pytorch torchvision torchaudio cpuonly -c pytorch
   ```
6. Install [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#installation-via-anaconda). Note: At the time of writing this, there are issues with installing PyG with conda. We used:
   ```
   pip install torch_geometric
   ```
7. Install [pandas](https://pandas.pydata.org/docs/getting_started/install.html#installing-from-pypi): ```pip install pandas```.
8. Install [matplotlib](https://matplotlib.org/stable/users/installing/index.html):
   ```
   python -m pip install -U pip
   python -m pip install -U matplotlib
   ```
