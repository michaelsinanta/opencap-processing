# %% Directories, paths, and imports. You should not need to change anything.
import os
import sys
import numpy as np
import pandas as pd

baseDir = os.getcwd()
opensimADDir = os.path.join(baseDir, 'UtilsDynamicSimulations', 'OpenSimAD')
sys.path.append(baseDir)
sys.path.append(opensimADDir)

from utilsOpenSimAD import processInputsOpenSimAD, plotResultsOpenSimAD
from mainOpenSimAD import run_tracking

def get_mot_time_range(mot_file_path):
    with open(mot_file_path, 'r') as f:
        # Skip header
        for line in f:
            if 'endheader' in line:
                break
        # Read the rest as data
        times = []
        for line in f:
            if line.strip() == '':
                continue
            time_str = line.split()[0]
            try:
                times.append(float(time_str))
            except ValueError:
                continue
        if times:
            return min(times), max(times)
        else:
            return None, None

# List to store paths of generated GRF files
grf_file_paths = []

def read_mot_file_to_df(file_path):
    """Reads a .mot file into a pandas DataFrame."""
    with open(file_path, 'r') as f:
        header_lines = []
        data_lines = []
        in_header = True
        for line in f:
            if in_header:
                header_lines.append(line)
                if 'endheader' in line:
                    in_header = False
            else:
                data_lines.append(line)
    
    # Extract column names from header (the line after 'endheader')
    column_names_line = [line for line in header_lines if 'endheader' in line][0]
    column_names_index = header_lines.index(column_names_line) + 1
    column_names = header_lines[column_names_index].strip().split('\t')

    # Read data into a DataFrame
    # Ensure data lines are not empty and contain numerical data
    valid_data_lines = [line.strip().split() for line in data_lines if line.strip() and not line.strip().startswith(';')] # Also ignore comment lines
    
    if not valid_data_lines:
        return pd.DataFrame(columns=column_names)

    df = pd.DataFrame(valid_data_lines, columns=column_names).astype(float)
    return df

def concatenate_grf_files(file_paths):
    """Reads multiple GRF .mot files and concatenates them into a single DataFrame."""
    all_grf_data = []
    for f_path in file_paths:
        try:
            df = read_mot_file_to_df(f_path)
            all_grf_data.append(df)
        except Exception as e:
            print(f"Error reading {f_path}: {e}")
            continue

    if not all_grf_data:
        return pd.DataFrame() # Return empty DataFrame if no data

    # Concatenate all DataFrames. Sort by time and drop duplicate time entries
    concatenated_df = pd.concat(all_grf_data, ignore_index=True)
    concatenated_df = concatenated_df.sort_values(by='time').drop_duplicates(subset=['time'])
    return concatenated_df


# User Inputs
session_id = "OpenCapData_ab7eb7cf-817d-4035-a30b-ee68773906cb"
trial_name = 'Suhasno_2'
session_type = 'overground'
motion_type = "walking"
repetition = None
# Set to True to solve the optimal control problem.
solveProblem = True
# Set to True to analyze the results of the optimal control problem. If you
# solved the problem already, and only want to analyze/process the results, you
# can set solveProblem to False and run this script with analyzeResults set to
# True. This is useful if you do additional post-processing but do not want to
# re-run the problem.
analyzeResults = True

# Path to where you want the data to be downloaded.
dataFolder = baseDir # Set dataFolder to the base directory

sessionFolder =  os.path.join(dataFolder, session_id)
pathTrial = os.path.join(sessionFolder, 'OpenSimData', 'Kinematics', trial_name + '.mot') 
start_time, end_time = get_mot_time_range(pathTrial)

if start_time is not None and end_time is not None:
    window_starts = np.arange(start_time, end_time, 1.0)
    window_ends = np.minimum(window_starts + 1.0, end_time)
    windows = list(zip(window_starts, window_ends))
else:
    windows = []

for i, (win_start, win_end) in enumerate(windows):
    current_time_window = [win_start, win_end]
    current_case = f'{trial_name}_window_{i}'

    print(f"Processing window: {current_time_window} with case: {current_case}")

    # %% Setup.
    if not 'repetition' in locals():
        repetition = None
    if not 'treadmill_speed' in locals():
        treadmill_speed = 0
    if not 'contact_side' in locals():
        contact_side = 'all'
    
    settings = processInputsOpenSimAD(baseDir, dataFolder, session_id, trial_name,
                                      motion_type, current_time_window, repetition,
                                      treadmill_speed, contact_side, use_local_data=True)
    
    # %% Simulation.
    run_tracking(baseDir, dataFolder, session_id, settings, case=current_case,
                  solveProblem=solveProblem, analyzeResults=analyzeResults)
    
    # Store the path to the generated GRF resultant file
    grf_file_paths.append(os.path.join(
        dataFolder, session_id, 'OpenSimData', 'DynamicSimulations', trial_name,
        f'GRF_resultant_{trial_name}_{current_case}.mot'))

# Concatenate all GRF files
print("Concatenating GRF files...")
full_grf_data = concatenate_grf_files(grf_file_paths)
print(f"Concatenated GRF data shape: {full_grf_data.shape}")

# Optional: Save the full GRF data to a new .mot file
output_grf_file = os.path.join(
    dataFolder, session_id, 'OpenSimData', 'DynamicSimulations', trial_name,
    f'GRF_resultant_{trial_name}_full.mot')

if not full_grf_data.empty:
    # Define helper function to write dataframe to .mot file format
    def write_df_to_mot(df, output_path):
        with open(output_path, 'w') as f:
            # Write header information (minimal for now, can be expanded)
            f.write(f'name {os.path.basename(output_path)}\n')
            f.write(f'datarows {len(df)}\n')
            f.write(f'datacolumns {len(df.columns)}\n')
            f.write(f'range {df['time'].min()} {df['time'].max()}\n')
            f.write('endheader\n')
            f.write('\t'.join(df.columns) + '\n')
            df.to_csv(f, sep='\t', index=False, header=False)

    write_df_to_mot(full_grf_data, output_grf_file)
    print(f"Full GRF data saved to: {output_grf_file}")
else:
    print("No GRF data to save.")