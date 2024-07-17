'''
Library containing various neural networks to be used for the prediction of inhalation times and sniff frequency from local field potentials (LFPs) recorded from the olfactory bulb of mice.


Author: Sid Rafilson
PI: Matt Smear
Lab: Smear Lab, University of Oregon

'''


import numpy as np
import pandas as pd
from typing import List, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import signal
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
import torch.profiler
import torch.cuda.amp as amp
import os
import time
import glob
import gc
import pyarrow.parquet as pq
import h5py
from tqdm import tqdm




#__________________________________________________________________Helper Functions and Classes__________________________________________________________________#


def load_sniff_MATLAB(file: str) -> np.array:
    '''
    Loads a MATLAB file containing sniff data and returns a numpy array
    '''

    mat = scipy.io.loadmat(file)
    sniff_params = mat['sniff_params']

    # loading sniff parameters
    inhalation_times = sniff_params[:, 0]
    inhalation_voltage = sniff_params[:, 1]
    exhalation_times = sniff_params[:, 2]
    exhalation_voltage = sniff_params[:, 3]



    # bad sniffs are indicated by 0 value in exhalation_times
    bad_indices = np.where(exhalation_times == 0, True, False)



    # removing bad sniffs
    inhalation_times = np.delete(inhalation_times, bad_indices)
    inhalation_voltage = np.delete(inhalation_voltage, bad_indices)
    exhalation_times = np.delete(exhalation_times, bad_indices)
    exhalation_voltage = np.delete(exhalation_voltage, bad_indices)

    return inhalation_times.astype(np.int32), inhalation_voltage, exhalation_times.astype(np.int32), exhalation_voltage



def sliding_window(data, window_size, step_size, stride_factor=1):
    '''
    Create sliding windows of the data with a specified stride factor

    Parameters
    ----------
    data : np.array
        Data to be windowed
    window_size : int
        Size of the window
    step_size : int
        Step size for the window
    stride_factor : int, optional
        Factor by which to stride within each window (default is 1)

    Returns
    -------
    np.array
        Array of windows

    np.array
        Array of indices in center of each window
    '''

    # Adjust window size for stride factor
    effective_window_size = (window_size - 1) * stride_factor + 1

    # Ensure the window size does not exceed the data length
    if len(data) < effective_window_size:
        raise ValueError("Window size with stride factor is larger than the data length")
    
    # Create the windows
    shape = ((data.size - effective_window_size) // step_size + 1, window_size)
    strides = (data.strides[0] * step_size, data.strides[0] * stride_factor)
    windows = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

    # Calculate indices of the middle points
    if window_size % 2 == 0:
        center_offset = (window_size // 2 - 1) * stride_factor + stride_factor // 2
    else:
        center_offset = (window_size // 2) * stride_factor

    indices = np.arange(center_offset, center_offset + shape[0] * step_size, step_size)

    return windows, indices



def calculate_inhalation(data):
    '''
    Calculate inhalation based on the start and end of inhalation

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the start and end of inhalation

    Returns
    -------
    pd.DataFrame
        Dataframe with inhalation column added
    '''

    inhalation_active = data['inhalation start'].cumsum() - data['inhalation end'].cumsum()
    data['inhalation'] = 0
    if np.min(inhalation_active) == -1:
        data.loc[inhalation_active > -1, 'inhalation'] = 1
    elif np.min(inhalation_active) == 0:
        data.loc[inhalation_active > 0, 'inhalation'] = 1
    else:
        print('Error in calculating inhalation')
    return data



def remove_bad_inhalations(data_current, count: int = 1000):
    '''
    Remove bad inhalations from the data which are all 1s or all 0s for more than 1 second and islands of data where the index is not continuous

    Parameters
    ----------
    data_current : pd.DataFrame
        Dataframe containing the data

    Returns
    -------
    pd.DataFrame
        Dataframe with bad inhalations removed
    '''

    # removing where inhalation is all 1s or all 0s for more than 1 second
    data_current['inh sum'] = data_current['inhalation'].rolling(count, min_periods = 1).sum()
    data_current['rev inh sum'] = data_current['inhalation'][::-1].rolling(count, min_periods = 1).sum()[::-1]
    bad_indicies = data_current[(data_current['inh sum'] == 0) | (data_current['inh sum'] == count) | (data_current['rev inh sum'] == 0) | (data_current['rev inh sum'] == count)].index
    data_current.drop(bad_indicies, inplace=True)
    data_current.drop(['inh sum', 'rev inh sum'], axis=1, inplace=True)

    # removing any islands of data where the index is not continuous
    diffs = np.diff(data_current.index.values)
    data_current['diff'] = np.concatenate([[1], diffs])
    edges = data_current[data_current['diff'] != 1].index
    for i in range(1, len(edges), 2):
        bad_indicies = np.concatenate([bad_indicies, np.arange(edges[i-1], edges[i])])
        data_current.drop(index = np.arange(edges[i-1], edges[i]), inplace=True, errors='ignore')
    data_current.drop(['diff'], axis=1, inplace=True)

    # resetting the index
    data_current.reset_index(drop=True, inplace=True)

    # sorting the indicies
    bad_indicies = np.sort(bad_indicies)


    return data_current, bad_indicies



def combine_parquet_files(dir):

    for data_type in ['target', 'feature']:
        if data_type == 'feature':
            file_name = 'processed'
        elif data_type == 'target':
            file_name = 'metadata'


        parquet_files = sorted(glob.glob(os.path.join(dir, f'*_{file_name}.parquet')))
        print(f"Loaded files: {parquet_files}")

        combined_df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)
        print(combined_df.shape)
        print(combined_df.head())

        combined_df.to_parquet(os.path.join(dir, f'data_{file_name}.parquet'), index=True)

        del combined_df



def get_data_length(file_path: str) -> int:
    parquet_file = pq.ParquetFile(file_path)
    return parquet_file.metadata.num_rows



class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss



class HDF5Dataset(Dataset):
    def __init__(self, h5_file, dataset_name, transform = None):
        self.h5_file = h5_file
        self.dataset_name = dataset_name
        self.transform = transform
        self.dataset = None

        with h5py.File(self.h5_file, 'r') as f:
            self.length = f[self.dataset_name].shape[0]

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.h5_file, 'r')[self.dataset_name]

        data = self.dataset[idx, :-1]
        target = self.dataset[idx, -1]

        sample = {'data': torch.tensor(data, dtype=torch.float32), 'target': torch.tensor(target, dtype=torch.float32)}

        if self.transform:
            sample = self.transform(sample)

        return sample

   
#__________________________________________________________________Models Definitions__________________________________________________________________#


class InhalationClassifierVanilla(nn.Module):
    def __init__(self, input_size):
        """
        Initializes the InhalationClassifierVanilla model.

        Parameters:
        - input_size (int): The size of the input features.

        """
        super(InhalationClassifierVanilla, self).__init__()
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

        # Activation function
        self.relu = nn.ReLU()

        # Batch normalization and dropout
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.batchnorm3 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.batchnorm1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.batchnorm3(x)
        x = self.dropout3(x)

        x = self.fc4(x)

        return x
    
 

class InhalationClassifierCNN(nn.Module):
    def __init__(self, input_size):
        """
        Initializes the InhalationClassifierCNN model.

        Parameters:
        - input_size (int): The size of the input features.

        Returns:
        - torch.Tensor: The output tensor.

        """
        super(InhalationClassifierCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=100, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=10, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=10, stride=1, padding=1)


        # Activation function
        self.relu = nn.ReLU()

        # Batch normalization and dropout
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.batchnorm3 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)

        # Pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)


        # Fully connected layers
        conv_output_size = self.calculate_conv_output_size(input_size)
        self.fc1 = nn.Linear(64 * conv_output_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.batchnorm4 = nn.BatchNorm1d(32)
        self.dropout4 = nn.Dropout(0.5)



    def calculate_conv_output_size(self, input_size):
        size = input_size
        size = (size - 100 + 2 * 1) // 1 + 1  # conv1
        size = size // 2  # pool1
        size = (size - 10 + 2 * 1) // 1 + 1  # conv2
        size = size // 2  # pool2
        size = (size - 10 + 2 * 1) // 1 + 1 # conv3
        size = size // 2 # pool3
        return size


    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.batchnorm1(x)
        x = self.dropout1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        x = self.dropout2(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.batchnorm3(x)
        x = self.dropout3(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.batchnorm4(x)
        x = self.dropout4(x)

        x = self.fc2(x)

        return x


#__________________________________________________________________Models Training and Testing__________________________________________________________________#

def train(train_data_loader, val_data_loader, model, loss_fn, scaler, accumulation_steps=4):


    train_start = time.time()

    model.train()
    optimizer.zero_grad()
    total_batches = 0 


    # Training
    for batch, batch_data in enumerate(tqdm(train_data_loader, desc='Training...')):
        X, y = batch_data['data'].to(device, non_blocking = True), batch_data['target'].to(device, non_blocking = True)

        with amp.autocast():
            pred = model(X)
            loss = loss_fn(pred, y.unsqueeze(1))

        # Normalize loss to account for gradient accumulation
        loss = loss / accumulation_steps

        # Backpropagation
        scaler.scale(loss).backward()

        # Step the optimizer and zero the gradients every accumulation_steps
        if (batch + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_batches += 1


        # Final step for any remaining gradients
        if total_batches % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()





            


    # Validation
    model.eval()
    val_loss, correct = 0, 0
    size = 0
    with torch.no_grad():
        for batch, batch_data in enumerate(tqdm(val_data_loader, desc='Validation...')):
            X, y = batch_data['data'].to(device, non_blocking = True), batch_data['target'].to(device, non_blocking = True)
            y = y.unsqueeze(1)

            pred = model(X)
            batch_loss = loss_fn(pred, y).item()
            val_loss += batch_loss
            correct += ((torch.sigmoid(pred) > 0.5) == y).type(torch.float).sum().item()
            size += y.size(0)

    avg_val_loss = val_loss / len(val_data_loader)
    accuracy = correct / size



    

    
    return avg_val_loss, accuracy



def test(data_loader, model, loss_fn, target: str = 'inhale'):
    model.eval()
    test_loss, correct = 0, 0
    size = 0

    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc='Testing...'):
            X, y = batch_data['data'].to(device, non_blocking = True), batch_data['target'].to(device, non_blocking = True)
            y = y.unsqueeze(1)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            if target == 'inhale':
                correct += ((torch.sigmoid(pred) > 0.5) == y).type(torch.float).sum().item()
            elif target == 'frequency':
                correct += ((pred - y) ** 2).sum().item()
            size += y.size(0)


    if size > 0:
        test_loss /= len(data_loader)
        correct /= size
    else:
        print("Warning: No samples processed in the test function. Please check the data loader.")
        test_loss = float('inf')
        correct = 0
    
    return test_loss, correct

#__________________________________________________________________Data Preprocessing__________________________________________________________________#


def load_data(mice: list[str] = ['4122', '4127', '4131', '4138'], sessions: Union[list[str], str] = 'all', nchannels: Union[int, list, str] = 2, path: str = r"E:\Sid_LFP\Sid_data\rnp_final", verbose: bool = False, plot_figs: bool = False, f: int = 1000, save_path: str = None):

    if verbose:
        start = time.time()
        print("Loading data\n-----------------------------------------")

    if save_path is None:
        raise ValueError("Save path must be specified.")


    # Create the save path if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Looping through the mice
    for mouse in mice:
        mouse_dir = os.path.join(path, mouse)

        if sessions == 'all':
            sessions_current = os.listdir(mouse_dir)
        elif isinstance(sessions, str):
            sessions_current = [sessions]
        else:
            sessions_current = sessions

        # Looping through the sessions for each mouse
        for session in sessions_current:
            if session not in os.listdir(mouse_dir):
                print(f"Session {session} not found for mouse {mouse}. Skipping...")
                continue
            session_dir = os.path.join(mouse_dir, session)
            if verbose:
                print(f"Loading data for mouse {mouse}, session {session}...")


            # Load the LFP and sniff signal data
            try:
                lfp = np.load(os.path.join(session_dir, 'LFP.npy'), mmap_mode='r')
                sniff_signal = scipy.io.loadmat(os.path.join(session_dir, 'sniff_signal.mat'))['sniff']
                inh, _, exh, _ = load_sniff_MATLAB(os.path.join(session_dir, 'sniff_params.mat'))

                if lfp.shape[1] != sniff_signal.shape[1]:
                    print(f"Length of LFP and sniff signal not equal for mouse {mouse}, session {session}. Skipping...")
                    continue

                inhalation = np.zeros(sniff_signal.shape[1])
                exhalation = np.zeros(sniff_signal.shape[1])
                inhalation[inh] = 1
                exhalation[exh] = 1

                diffs = np.diff(inh)
                freqs = f / diffs
                frequencies = np.zeros(sniff_signal.shape[1])
                for i in range(len(inh) - 1):
                    frequencies[inh[i]:inh[i+1]] = freqs[i]

                if isinstance(nchannels, int):
                    lfp = lfp[:nchannels, :]
                elif nchannels == 'all':
                    pass
                elif isinstance(nchannels, list):
                    lfp = lfp[nchannels, :]
                else:
                    raise ValueError("Invalid value for nchannels. Must be an integer, list, or 'all'.")

                sniff_signal = sniff_signal[0, :].flatten()
                inh_start = inhalation.flatten()
                inh_end = exhalation.flatten()


                # Building the dataframe to save the data
                current_data = pd.DataFrame({
                    'sniff_signal': sniff_signal,
                    'inhalation start': inh_start,
                    'inhalation end': inh_end,
                    'frequency': frequencies,
                })

                for channel in range(lfp.shape[0]):
                    current_data[f'channel_{channel}'] = lfp[channel, :]

                current_data = calculate_inhalation(current_data)
                current_data['time'] = np.arange(0, current_data.shape[0], 1)


                # Save the data
                session_save_path = os.path.join(save_path, f"{mouse}_{session}_data.parquet")
                current_data.to_parquet(session_save_path)

                # Plot the figures
                if plot_figs:
                    fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True, sharey=False)
                    axs[0].scatter(current_data['time'][current_data['inhalation start'] == 1], current_data['sniff_signal'][current_data['inhalation start'] == 1], c='r')
                    axs[0].scatter(current_data['time'][current_data['inhalation end'] == 1], current_data['sniff_signal'][current_data['inhalation end'] == 1], c='g')
                    sns.scatterplot(data=current_data, x='time', y='sniff_signal', ax=axs[0], hue='inhalation')
                    sns.lineplot(data=current_data, x='time', y='frequency', ax=axs[1])
                    sns.lineplot(data=current_data, x='time', y='channel_0', ax=axs[2])
                    plt.show()

                print(f"Data for mouse {mouse}, session {session} saved to {session_save_path}")

            except Exception as e:
                print(f"Error loading data for mouse {mouse}, session {session}: {e}")
                continue

    if verbose:
        print("Data loaded.")
        end = time.time()
        print(f"Time taken: {end - start} seconds.\n")



def process_data_windows(data_dir: str, filter: tuple = (None, None), window_size: int = 1000, step_size: int = 1, stride_factor: int = 10, verbose: bool = False, f: int = 1000):
    
    if verbose:
        start = time.time()
        print("Processing data\n-----------------------------------------")

    # Loop through the session files created in the load_data function
    session_files = [file for file in os.listdir(data_dir) if file.endswith('data.parquet')]
    for session_file in tqdm(session_files, desc='Processing data...', total=len(session_files)):


        try:
            # Load the data
            data_current = pd.read_parquet(os.path.join(data_dir, session_file))

            # Get the mouse and session
            mouse = session_file.split('_')[0]
            session = session_file.split('_')[1].split('.')[0]
            
            # Get the channels
            channels_current = [col for col in data_current.columns if 'channel' in col]
            if not channels_current:
                tqdm.write(f"No channel data found for mouse {mouse}, session {session}. Skipping...")
                continue
            else:
                if verbose:
                    tqdm.write(f"Processing mouse {mouse}, session {session}, and channels {channels_current}...")


            # Process the LFP data
            all_current_windows = []
            all_current_indices = []
            for ch in channels_current:
                current_lfp = data_current[ch].values
                del data_current[ch]


                if filter[0] is not None and filter[1] is not None:
                    sos = signal.butter(10, filter, 'bandpass', fs=f, output='sos')
                    current_lfp = signal.sosfiltfilt(sos, current_lfp)
                elif filter[0] is not None:
                    sos = signal.butter(10, filter[0], 'highpass', fs=f, output='sos')
                    current_lfp = signal.sosfiltfilt(sos, current_lfp)
                elif filter[1] is not None:
                    sos = signal.butter(10, filter[1], 'lowpass', fs=f, output='sos')
                    current_lfp = signal.sosfiltfilt(sos, current_lfp)


                if len(current_lfp) < window_size:
                    tqdm.write(f"Window size is larger than the data length for mouse {mouse}, session {session}. Skipping...")
                    continue
                
                # Create the windows of LFP data
                windows, indices = sliding_window(current_lfp, window_size, step_size, stride_factor=stride_factor)
                del current_lfp

                valid_indices = indices[indices < len(data_current)]
                valid_windows = windows[:len(valid_indices)]
                del windows, indices

                all_current_windows.append(valid_windows)
                all_current_indices.append(valid_indices)
                del valid_windows, valid_indices

                gc.collect()

            if not all_current_indices:
                continue


            # Concatenate the windows and indices
            all_current_windows = np.concatenate(all_current_windows, axis=1)
            all_current_indices = np.concatenate(all_current_indices)

 
            all_current_indices = np.unique(all_current_indices)
            data_current = data_current.iloc[all_current_indices]
            data_current.reset_index(drop=True, inplace=True)


            # Remove the inhalations that are all 1s or all 0s for more than 1 second and islands of data where the index is not continuous
            data_current, removed_indices = remove_bad_inhalations(data_current, 1000)
            all_current_windows = np.delete(all_current_windows, removed_indices, axis=0)



            # Save the processed data
            processed_save_path = os.path.join(data_dir, f"{mouse}_{session}_processed.parquet")
            pd.DataFrame(all_current_windows).to_parquet(processed_save_path, index=False)

            # Save the metadata
            meta_data_save_path = os.path.join(data_dir, f"{mouse}_{session}_metadata.parquet")
            data_current.to_parquet(meta_data_save_path)


            # free up memory
            del all_current_windows, all_current_indices, data_current, removed_indices
            gc.collect()


        except Exception as e:
            print(f"Error processing data for session file {session_file}: {e}")
            continue



    if verbose:
        end = time.time()
        print(f"Time taken: {(end - start) / 60} minutes.\n")



def scale_data(processing_dir: str):
    scaler = StandardScaler()

    # Loading the processed data and fitting the scaler one session at a time
    files = [file for file in os.listdir(processing_dir) if file.endswith('_processed.parquet')]
    print('\nFitting scaler\n-----------------------------------------')
    for file in tqdm(files, desc='Fitting scaler...'):
        if file.endswith('_processed.parquet'):
            scaler.partial_fit(pd.read_parquet(os.path.join(processing_dir, file)))


    # Scaling the data one session at a time
    print('\nScaling data\n-----------------------------------------')
    for file in tqdm(files, desc='Scaling data...'):
        if file.endswith('_processed.parquet'):
            X_scaled = scaler.transform(pd.read_parquet(os.path.join(processing_dir, file)))

            # Saving the scaled data
            mouse = file.split('_')[0]
            session = file.split('_')[1].split('_')[0]
            scaled_save_path = os.path.join(processing_dir, f"{mouse}_{session}_scaled.parquet")
            pd.DataFrame(X_scaled).to_parquet(scaled_save_path, index=False)



def split_data(data_dir: str, target: str = 'inhalation', chunk_size: int = 100000, verbose: bool = True, test_session = None):
    
    
    start = time.time()
    print("\nSplitting data...\n-----------------------------------------")
    
    # List all the files
    features_files = sorted([file for file in os.listdir(data_dir) if file.endswith('scaled.parquet')])
    targets_files = sorted([file for file in os.listdir(data_dir) if file.endswith('metadata.parquet')])
    if verbose:
        print(f"Features files: {features_files}\nTargets files: {targets_files}")

    # Open HDF5 file for writing
    with h5py.File(os.path.join(data_dir, 'data.h5'), 'w') as f:
        train_data = f.create_dataset('train_data', (0, 1001), maxshape=(None, 1001), compression='gzip', chunks=True)
        val_data = f.create_dataset('val_data', (0, 1001), maxshape=(None, 1001), compression='gzip', chunks=True)
        test_data = f.create_dataset('test_data', (0, 1001), maxshape=(None, 1001), compression='gzip', chunks=True)

        # Read and process data in chunks
        for i, (features_file, targets_file) in enumerate(zip(features_files, targets_files)):

            mouse = features_file.split('_')[0]
            session = features_file.split('_')[1].split('_')[0]
            print(f"Processing data for mouse {mouse}, session {session}...")

            features = pd.read_parquet(os.path.join(data_dir, features_file))
            targets = pd.read_parquet(os.path.join(data_dir, targets_file))[target]

            # Shuffle the data
            perm = np.random.permutation(len(features))
            features = features.iloc[perm]
            targets = targets.iloc[perm]

            # Process in smaller chunks to save memory
            for start_idx in tqdm(range(0, len(features), chunk_size), desc=f'Processing chunks for {features_file}...'):
                end_idx = min(start_idx + chunk_size, len(features))
                chunk_features = features.iloc[start_idx:end_idx]
                chunk_targets = targets.iloc[start_idx:end_idx]



                # Handeling when test_session is None. Train, Test, and Val split all in one go
                if test_session is None:
                    train_features, test_features, train_target, test_target = train_test_split(chunk_features, chunk_targets, test_size=0.2, random_state=42)
                    train_features, val_features, train_target, val_target = train_test_split(train_features, train_target, test_size=0.2, random_state=42)

                    # Convert to numpy and concatenate features and target
                    train_chunk = np.hstack((train_features.to_numpy(), train_target.to_numpy().reshape(-1, 1)))
                    val_chunk = np.hstack((val_features.to_numpy(), val_target.to_numpy().reshape(-1, 1)))
                    test_chunk = np.hstack((test_features.to_numpy(), test_target.to_numpy().reshape(-1, 1)))

                    # Append to HDF5 datasets
                    for dataset, chunk in zip([train_data, val_data, test_data], [train_chunk, val_chunk, test_chunk]):
                        dataset.resize(dataset.shape[0] + chunk.shape[0], axis=0)
                        dataset[-chunk.shape[0]:] = chunk

                    # cleaning up
                    del train_features, val_features, test_features, train_target, val_target, test_target



                # Handeling when test_session is not None and we are on the test session
                elif test_session == session:
                    test_features, test_target = chunk_features, chunk_targets

                    # Convert to numpy and concatenate features and target
                    test_chunk = np.hstack((test_features.to_numpy(), test_target.to_numpy().reshape(-1, 1)))

                    # Append to HDF5 datasets
                    test_data.resize(test_data.shape[0] + test_chunk.shape[0], axis=0)
                    test_data[-test_chunk.shape[0]:] = test_chunk

                    # cleaning up
                    del test_features, test_target


    
                # Handeling when test_session is not None and we are not on the test session
                elif test_session != session:
                    train_features, val_features, train_target, val_target = train_test_split(chunk_features, chunk_targets, test_size=0.2, random_state=42)

                    # Convert to numpy and concatenate features and target
                    train_chunk = np.hstack((train_features.to_numpy(), train_target.to_numpy().reshape(-1, 1)))
                    val_chunk = np.hstack((val_features.to_numpy(), val_target.to_numpy().reshape(-1, 1)))

                    # Append to HDF5 datasets
                    for dataset, chunk in zip([train_data, val_data], [train_chunk, val_chunk]):
                        dataset.resize(dataset.shape[0] + chunk.shape[0], axis=0)
                        dataset[-chunk.shape[0]:] = chunk

                    # cleaning up
                    del train_features, val_features, train_target, val_target


                else:
                    raise ValueError("Invalid value for test_session.")

                



    print(f"Data split and saved to {os.path.join(data_dir, 'data.h5')}\nTotal time taken: {(time.time() - start) / 60} minutes.")



def load_datasets(data_dir: str, batch_size: int = 4096, num_workers: int = 16, prefetch_factor: int = 16):
    
    train_dataset = HDF5Dataset(os.path.join(data_dir, 'data.h5'), 'train_data')
    val_dataset = HDF5Dataset(os.path.join(data_dir, 'data.h5'), 'val_data')
    test_dataset = HDF5Dataset(os.path.join(data_dir, 'data.h5'), 'test_data')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, prefetch_factor=prefetch_factor, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, prefetch_factor=prefetch_factor, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, prefetch_factor=prefetch_factor, pin_memory=True, shuffle=False)
    
    return train_loader, val_loader, test_loader



#___________________________________________________________________Profiler___________________________________________________________________#

def profile_data_loading(data_loader, num_batches=100):
    start_time = time.time()
    for i, batch in enumerate(tqdm(data_loader, desc='Profiling Data Loading')):
        if i >= num_batches:
            break
        data, target = batch['data'], batch['target']
    end_time = time.time()
    print(f"Time taken for {num_batches} batches: {end_time - start_time} seconds")



#__________________________________________________________________Workflow________________________________________________________________________________________________________________________________#

def inhalation_classifier_workflow():
    
    global optimizer, device, log_dir

    load = False
    process = False
    scale = False
    split = True

    

              
    window_size = 1000
    stride_factor = 1
    step_size = 1
    initial_lr = 1e-3
    batch_size = 8192
    accumulation_steps = 32
    num_workers = 16
    prefetch_factor = 4



    processing_dir = r"C:\data\networks\4127_ch1_nofilt_test5"
    os.makedirs(processing_dir, exist_ok=True)

    log_dir = r"E:\Sid_LFP\PyTorch Network\log\profiler"
    os.makedirs(log_dir, exist_ok=True)

    save_dir = r"E:\Sid_LFP\PyTorch Network\models\4127\ch1_nofilt_test5"
    os.makedirs(save_dir, exist_ok=True)

    # Creating a txt file with the parameters
    with open(os.path.join(save_dir, 'parameters.txt'), 'w') as f:
        f.write(f"Window size: {window_size}\n")
        f.write(f"Stride factor: {stride_factor}\n")
        f.write(f"Step size: {step_size}\n")
        f.write(f"Initial learning rate: {initial_lr}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Accumulation steps: {accumulation_steps}\n")
        f.write(f"Number of workers: {num_workers}\n")
        f.write(f"Prefetch factor: {prefetch_factor}\n")
        f.write(f"Processing directory: {processing_dir}\n")
        f.write(f"Log directory: {log_dir}\n")
        f.write(f"Save directory: {save_dir}\n")


 


    if load:
        load_data(mice = ['4122'], sessions = 'all' , verbose = True, nchannels = 1, path = r"E:\Sid_LFP\Sid_data\rnp_final", save_path = processing_dir)

    if process:
        process_data_windows(processing_dir, filter = (None, None), window_size = window_size, step_size = step_size, stride_factor = stride_factor, verbose = True, f = 1000)

    if scale:
        scale_data(processing_dir)
    
    if split:
        split_data(processing_dir, target = 'inhalation', verbose = True, test_session = '5')

    


    
    data_dir = processing_dir

    # Loading the data
    print('\nLoading the data...')
    start = time.time()
    train_loader, val_loader, test_loader = load_datasets(data_dir, batch_size = batch_size, num_workers = num_workers, prefetch_factor = prefetch_factor)
    print(f"\nData Loaded.\nTime elapsed: {(time.time() - start)} seconds\nLength of train loader: {len(train_loader)} Length of val loader: {len(val_loader)} Length of test loader: {len(test_loader)}\n")


    # Setting the device to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    with open(os.path.join(save_dir, 'parameters.txt'), 'a') as f:
        f.write(f"Device: {device}\n")



    # Initialize the model, loss function, and optimizer
    model = InhalationClassifierCNN(window_size).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    print(f"Model: {model}\nLoss function: {loss_fn}\nOptimizer: {optimizer}\n")
    with open(os.path.join(save_dir, 'parameters.txt'), 'a') as f:
        f.write(f"Model: {model}\nLoss function: {loss_fn}\nOptimizer: {optimizer}\n")


    # Add a learning rate scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    with open(os.path.join(save_dir, 'parameters.txt'), 'a') as f:
        f.write(f"Scheduler: {scheduler}\n")

    # Add early stopping
    earlystopping = EarlyStopping(patience=10, verbose=True)
    with open(os.path.join(save_dir, 'parameters.txt'), 'a') as f:
        f.write(f"Early stopping: {earlystopping}\n")

    # Initialize a GradScaler for mixed precision training
    scaler = amp.GradScaler()
    print(f"Scaler: {scaler}\n")
    with open(os.path.join(save_dir, 'parameters.txt'), 'a') as f:
        f.write(f"Scaler: {scaler}\n\n\n")


    # Train the model
    epochs = 3
    val_losses = []
    val_accuracies = []
    full_train_start = time.time()
    for t in range(epochs):
        start = time.time()
        print(f"\nEpoch {t+1}\n-------------------------------")
        with open(os.path.join(save_dir, 'parameters.txt'), 'a') as f:
            f.write(f"\nEpoch {t+1}\n-------------------------------\n")

        # Training the model
        loss, accuracy = train(train_loader, val_loader, model, loss_fn, scaler, accumulation_steps=accumulation_steps)
        val_losses.append(loss)
        val_accuracies.append(accuracy)
        print(f"Training time: {(time.time() - start) / 60} minutes\nTraining Error: \n Avg loss: {loss:>8f} \n Accuracy: {accuracy:>8f}")
        with open(os.path.join(save_dir, 'parameters.txt'), 'a') as f:
            f.write(f"Training time: {(time.time() - start) / 60} minutes\nTraining Error: \n Avg loss: {loss:>8f} \n Accuracy: {accuracy:>8f}\n")

        # Step the learning rate scheduler
        scheduler.step()

        # Early stopping
        earlystopping(loss, model)
        if earlystopping.early_stop:
            print("Early stopping")
            break

    print(f"Done\n-----------------------------------------\nTotal training time: {(time.time() - full_train_start) / 60} minutes\n")
    with open(os.path.join(save_dir, 'parameters.txt'), 'a') as f:
        f.write(f"Done\n-----------------------------------------\nTotal training time: {(time.time() - full_train_start) / 60} minutes\n")

    loss, accuracy = test(test_loader, model, loss_fn, target = 'inhale')
    print(f'Testing performance\n------------------------Loss = {loss}\nAccuracy = {accuracy}')
    with open(os.path.join(save_dir, 'parameters.txt'), 'a') as f:
        f.write(f'Testing performance\n------------------------Loss = {loss}\nAccuracy = {accuracy}')

    # Plot the validation loss and accuracy
    val_losses = np.array(val_losses)
    val_accuracies = np.array(val_accuracies)
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    sns.despine()
    axs[0].plot(val_losses, label='Validation loss', color = 'dodgerblue')
    axs[0].legend()
    axs[1].plot(val_accuracies, label='Validation accuracy', color = 'dodgerblue')
    axs[1].legend()

  
    plt.savefig(os.path.join(save_dir, 'val_loss_accuracy.png'))




        
if __name__ == '__main__':

    inhalation_classifier_workflow()


    
