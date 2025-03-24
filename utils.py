import numpy as np
import torch
from torch_geometric.data import Data
import pandas as pd


# Create the graph data
def create_graph_data(start_coords, end_coords, attributes, edge_weight=None):
    data_list = []
    for i in range(len(start_coords)):
        start = torch.tensor(np.transpose(start_coords[i]), dtype=torch.float)
        attr = torch.tensor(np.repeat([attributes[i]], start.shape[0], axis=0), dtype=torch.float)
        end = np.transpose(end_coords[i])
        # Create the graph of 15 nodes with all to all connections
        edge_index = torch.tensor([[k, l] for k in range(16) for l in range(16) if k!=l], dtype=torch.long)
        # compute edge weights as the inverse of the distance between start[:,0] and start[:,1]
        if edge_weight is None:
            edge_weight = torch.tensor([0.1/torch.sqrt(torch.norm(start[edge_index[k]][0]-start[edge_index[k]][1])) for k in range(edge_index.shape[0])], dtype=torch.float)
        else:
            edge_weight = edge_weight.clone().detach()
        x = torch.concat((start, attr), dim=1)
        y = torch.tensor(end, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y, edge_attr=edge_weight)
        data_list.append(data)
    return data_list

# Parse the excel file
def parse_excel_train(file_path):
    df = pd.read_excel(file_path)
    patients = df['ID'].unique()
    # Take only the columns with the coordinates
    df_coords = df.iloc[:, 3:]
    # Take only the odd columns and convert to numpy array
    y_coords = df_coords.iloc[:, 1::2].to_numpy(dtype=np.float64)
    # Take only the even columns and convert to numpy array
    x_coords = df_coords.iloc[:, ::2].to_numpy(dtype=np.float64)
    # Add two colums to the dataframe
    df['x_coords'] = [x_coords[i] for i in range(x_coords.shape[0])]
    df['y_coords'] = [y_coords[i] for i in range(y_coords.shape[0])]
    start_coords = []
    end_coords = []
    attributes = []
    for patient in patients:
        # Take all the rows with the same patient
        patient_df = df[df['ID'] == patient]
        # Take all the possible couple of rows with ordered 'AGE' column
        patient_df = patient_df.sort_values('AGE')
        # Iterate on rows
        for i in range(len(patient_df) - 1):
            for j in range(i + 1, len(patient_df)):
                # Take the x and y coordinates of the two rows
                x1 = patient_df.iloc[i]['x_coords']
                y1 = patient_df.iloc[i]['y_coords']
                x2 = patient_df.iloc[j]['x_coords']
                y2 = patient_df.iloc[j]['y_coords']
                # compute age difference
                age_diff = patient_df.iloc[j]['AGE']- patient_df.iloc[i]['AGE']
                # create new dataframe with the new columns
                age = patient_df.iloc[i]['AGE']
                is_female = patient_df.iloc[i]['IS_FEMALE']
                start_coords.append((x1, y1))
                end_coords.append((x2, y2))
                attributes.append((age, age_diff, is_female))
    # Convert to numpy array and normalize
    mean = np.transpose(np.array([[35, -75]] * 16))
    var = np.transpose(np.array([[65, 75]] * 16))
    start_coords = (np.array(start_coords) - mean) / var
    end_coords = (np.array(end_coords) - mean) / var
    attributes = np.array(attributes) / np.array([20, 10, 1])

    return start_coords, end_coords, attributes

def parse_excel_test(file_path):
    df = pd.read_excel(file_path)
    df_coords = df.iloc[:, 4:]
    # Take only the odd columns and convert to numpy array
    y_coords = df_coords.iloc[:, 1::2].to_numpy(dtype=np.float64)
    # Take only the even columns and convert to numpy array
    x_coords = df_coords.iloc[:, ::2].to_numpy(dtype=np.float64)
    coords = np.stack([x_coords, y_coords],axis=1)
    # compute age difference
    age_diff = df.iloc[:,3].to_numpy(dtype=np.float64)
    # create new dataframe with the new columns
    age = df.iloc[:,2].to_numpy(dtype=np.float64)
    is_female = df.iloc[:,1].to_numpy(dtype=np.float64)
    attributes = [age, age_diff, is_female]
    mean = np.transpose(np.array([[35, -75]] * 16))
    var = np.transpose(np.array([[65, 75]] * 16))
    coords = (np.array(coords) - mean) / var
    attributes = np.array(attributes).transpose() / np.array([20, 10, 1])
    return coords, attributes

def output_to_excel(predicted_coords, file_path):
    columns = ['CoX', 'CoY', 'ArX', 'ArY', 'GoX', 'GoY', 'MeX',	'MeY', 'GnX', 'GnY', 'PgX', 'PgY', 'BX', 'BY', 'AX', 'AY',
               'ANSX', 'ANSY', 'PNSX', 'PNSY', 'NX', 'NY', 'SX', 'SY', 'PoX', 'PoY', 'OrX', 'OrY', 'BaX', 'BaY', 'PT Point X', 'PT Point Y']
    mean = np.transpose(np.array([[35, -75]] * 16))
    var = np.transpose(np.array([[65, 75]] * 16))
    data = []
    for coords in predicted_coords:
        coords = coords*var[:, 0] + mean[:, 0]
        unraveled_coords = []
        for i in range(coords.shape[0]):
            unraveled_coords.append(coords[i, 0])
            unraveled_coords.append(coords[i, 1])
        data.append(unraveled_coords)
    df = pd.DataFrame(data, columns=columns)
    # Save the dataframe to an excel file
    df.to_excel(file_path, index=False)
    return