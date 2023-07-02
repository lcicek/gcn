# Explaining and approximating a graph convolutional network trained on the IMDB-BINARY dataset
The GCN trained on IMDB-BINARY achieves an accuracy of 0.74. Using a custom graph class ```graph.py``` to access implicit properties in graphs from the IMDB-BINARY dataset enables an explanation and approximation of the GCN model (with an accuracy of up to 0.84) as well as a direct classification accuracy of 0.71 on the dataset.

## Setup
### Requirements
This project was run with Python 3.11.3 under Windows 10 (64-bit). Dependencies and exact package versions are listed in requirements.txt.
### Installation
1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Create conda environment with:
   ```conda create --name imdb-gcn python=3.11.3```
   Note: On Windows you can enter all of these commands in the Anaconda prompt (see [conda docs](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html#starting-conda)).
4. Activate conda environment: ```conda activate imdb-gcn```
5. Install [PyTorch](https://pytorch.org/) in activated conda environment. At the time of writing, we used the following command to install PyTorch version 2.0.1:
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
9. Optional: Change to project folder and open environment in Visual Studio Code by typing the command ```code``` in the Anaconda Prompt.
## Documentation
### Results
To replicate our results, run one of the following commands:
1. To see the model accuracy, run: ```python main.py -evalFinal```
2. To see the approximation accuracies, run: ```python approximation.py```
### Training
To train a new model, run: ```python main.py -train```
### Test
To visualize and/or modify graphs, change working directory to project directory and run: ```python test.py```
Visualized graphs are saved under images/graph.png. For more information please read instructions in ```test.py```.
### Custom graph class
Opposed to IMDB-BINARY dataset graphs, the custom graphs also contain:
- Node features ```x``` containing the degree of nodes.
- Edges ```edges``` of shape [2, num_edges].
- A list of ```movies``` implicit in the graph. Each movie contains all actors with an edge between each other.
- Functions ```removeActors()```, ```addActorsToMovie()``` and ```removeEdge()``` to modify graphs.
The custom graph class can be used for understanding and explanation of the dataset and/or model.
