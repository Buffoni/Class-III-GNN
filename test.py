import torch
from model import GNN
import numpy as np
from utils import create_graph_data, parse_excel_test, output_to_excel
import os

data_path = './test_data_sample.xlsx'
model_path = './model.pth'

if __name__ == '__main__':
    # Parse the excel file
    start_coords, attributes = parse_excel_test(data_path)

    # Create the graph
    data_list = create_graph_data(start_coords, start_coords, attributes)

    # Load the model from state dict
    model = GNN(data_list[0])
    model.load_state_dict(torch.load(model_path))
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
    output_path = os.path.join(os.path.dirname(data_path), f'predicted_coords.xlsx')
    output_to_excel(to_save, output_path)