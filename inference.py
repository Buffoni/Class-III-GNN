import torch
from model import GNN
import numpy as np
from utils import create_graph_data, parse_excel_test, output_to_excel
import os

def process_file(file_path, num_value):
    # Load the data
    data_path = file_path
    otuput_path = os.path.join(os.path.dirname(file_path),'predicted_coords.xlsx')

    # Parse the excel file
    start_coords, attributes = parse_excel_test(data_path)
    attributes[:,1] = num_value / 10
    # Create the graph
    data_list = create_graph_data(start_coords, start_coords, attributes)

    # Load the model from state dict
    model = GNN(data_list[0])
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    # Generate the predictions
    to_save = []
    with torch.no_grad():
        for j, data in enumerate(data_list):
            preds = []
            for i in range(10):
                x, attr = torch.split(data.x, [2, 3], dim=1)
                preds.append(model(data).numpy() + x.numpy())
            to_save.append(np.mean(preds, axis=0))

    # Save the predictions
    output_to_excel(to_save, otuput_path)
    mean = np.transpose(np.array([[35, -75]] * 16))
    var = np.transpose(np.array([[65, 75]] * 16))
    # Create a new dataframe with the predicted coordinates and attributes
    to_plot = np.mean(preds,axis=0) * var[:, 0] + mean[:, 0]
    start_coords = np.array(start_coords[0].transpose()) * var[:, 0] + mean[:, 0]
    return [start_coords, to_plot]