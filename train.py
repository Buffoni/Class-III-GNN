import torch
import torch.nn as nn
import random
import numpy as np
from model import GNN
from torch.optim.lr_scheduler import StepLR
from utils import create_graph_data, parse_excel_train
import datetime
import os

data_path = './train_data_sample.xlsx'

if __name__ == '__main__':
    # Parse the excel file
    start_coords, end_coords, attributes = parse_excel_train(data_path)

    # Create the graph
    data_list = create_graph_data(start_coords, end_coords, attributes)

    # Train the model
    model = GNN(data_list[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3,weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
    criterion = nn.MSELoss(reduction='sum')
    model.train()
    num_epochs = 100
    for epoch in range(num_epochs):
        random.shuffle(data_list)
        losses = []
        for data in data_list:
            optimizer.zero_grad()
            out = model(data)
            x, attr = torch.split(data.x, [2, 3], dim=1)
            out = out + x # the model output is the predicted shift of the initial coordinates
            loss = criterion(out,data.y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f'Epoch {epoch}, Loss: {np.mean(losses)}')

    # Save the model with a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(os.path.dirname(data_path),f'model_{timestamp}.pth')
    torch.save(model.state_dict(), out_file)