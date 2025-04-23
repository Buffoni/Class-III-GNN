![Project Logo](logo.png) 
# Graph Neural Network for Class III Malocclusion Prediction

## Overview

This project implements a Graph Neural Network (GNN) to analyze patient data. The GNN is built using PyTorch and PyTorch Geometric libraries. The project includes utilities for parsing patient data from Excel files, training the GNN model, and outputting predictions to Excel files.

## Project Structure

- `utils.py`: Contains utility functions for parsing Excel files and preparing data for the GNN.
- `model.py`: Defines the GNN model using PyTorch and PyTorch Geometric the pretrained weights of the model are in `model.pth`.
- `main.py`: The main script to run the pretrained GNN model to predict new data with a simple GUI.
- `test.py`: Contains the script to process in bulk the predictions of several patients using the pretrained model.
- `train.py`: Contains the script to train the GNN model given some training data.

## Installation

1. Clone or download the repository:
    ```sh
    git clone https://github.com/Buffoni/Class-III-GNN.git
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To open a simple GUI to predict new data using the pretrained model, run:

```sh
python main.py
```

The script plots the predictions of the patient provided in and saves them as an image and as a new set of cefalometric coordinates in Excel

One can also create an executable file using PyInstaller to run the GUI without needing Python installed. To do this, run:

```sh
pyinstaller --onefile --windowed --add-data "inference.py:." --add-data "utils.py:." main.py
```

### Parsing Excel Files

For the GNN to work, you need to parse the patient data from Excel files. The `utils.py` file contains functions to read the data and prepare it for the model.
The Excel files for training and testing should be formatted in the same way as the example files provided in the repository.

## License

This project is licensed under the MIT License.


## Cite
If you use this code in your research, please cite the following paper:

```
@article{learningmalocclusion,
  title={Machine Learning-Based model for predicting short- and long-term growth in untreated Class III malocclusion},
  author={Maria Denisa Statie, Michele Nieri, Valentina Rutili, Lorenzo Buffoni, Lorenzo Chicchi, Pietro Auconi, James A. McNamara, Lorenzo Franchi},
  journal={TBD},
  year={2025},
  volume={TBD},
  pages={TBD}
}
```
