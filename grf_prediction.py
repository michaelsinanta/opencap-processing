# %% Directories, paths, and imports. You should not need to change anything.
import os
import sys
import numpy as np
import pandas as pd
import multiprocessing
import argparse

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

def process_single_window(baseDir, dataFolder, session_id, trial_name, motion_type,
                          current_time_window, repetition, treadmill_speed, contact_side,
                          solveProblem, analyzeResults, case_prefix, window_index):

    current_case = f'{case_prefix}_window_{window_index}'
    print(f"Processing window: {current_time_window} with case: {current_case}")

    grf_file_path = None
    optimaltrajectories_file_path = None

    try:
        # %% Setup.
        # These variables are passed as arguments now.
        settings = processInputsOpenSimAD(baseDir, dataFolder, session_id, trial_name,
                                          motion_type, current_time_window, repetition,
                                          treadmill_speed, contact_side, use_local_data=True)

        # %% Simulation.
        run_tracking(baseDir, dataFolder, session_id, settings, case=current_case,
                      solveProblem=solveProblem, analyzeResults=analyzeResults)

        potential_grf_path = os.path.join(
            dataFolder, session_id, 'OpenSimData', 'DynamicSimulations', trial_name,
            f'GRF_resultant_{trial_name}_{current_case}.mot')
        if os.path.exists(potential_grf_path):
            grf_file_path = potential_grf_path
        else:
            print(f"Warning: GRF file not found for window {current_time_window} after simulation.")

        optimaltrajectories_file_path = os.path.join(
            dataFolder, session_id, 'OpenSimData', 'DynamicSimulations', trial_name,
            f'optimaltrajectories_{current_case}.npy')

        if not os.path.exists(optimaltrajectories_file_path):
            print(f"Warning: Optimal trajectories NPY file not found for window {current_time_window} after simulation. Expected at {optimaltrajectories_file_path}")
            optimaltrajectories_file_path = None

    except Exception as e:
        print(f"Error processing window {current_time_window} for trial {trial_name}: {e}")
        grf_file_path = None
        optimaltrajectories_file_path = None

    return grf_file_path, optimaltrajectories_file_path

def concatenate_npy_files(file_paths):
    if not file_paths:
        return {}

    all_data = []
    for f_path in file_paths:
        try:
            data = np.load(f_path, allow_pickle=True).item()
            all_data.append(data)
        except Exception as e:
            print(f"Error reading {f_path}: {e}")
            continue

    if not all_data:
        return {}

    # Initialize combined_data with the first dictionary
    combined_data = all_data[0].copy()

    # Concatenate subsequent dictionaries
    for i in range(1, len(all_data)):
        current_data = all_data[i]
        for key, value in current_data.items():
            if key in combined_data:
                # Concatenate numpy arrays along the first axis (time)
                if isinstance(value, np.ndarray) and isinstance(combined_data[key], np.ndarray):
                    combined_data[key] = np.concatenate((combined_data[key], value), axis=0)
                # Handle other types if necessary, e.g., lists, scalars
                elif isinstance(value, list) and isinstance(combined_data[key], list):
                    combined_data[key].extend(value)
                elif isinstance(value, (int, float, str)):
                    # If it's a scalar, we'll keep the last one or log a warning.
                    # For now, let's just keep the last one, assuming it's consistent or a single value.
                    pass # Or you could choose to overwrite: combined_data[key] = value
                else:
                    print(f"Warning: Cannot concatenate data for key '{key}' of type {type(value)}. Keeping data from last file.")
                    # For now, if types are inconsistent or cannot be concatenated, just keep the last one
                    combined_data[key] = value
            else:
                combined_data[key] = value

    return combined_data

def main():
    parser = argparse.ArgumentParser(description='Process OpenCap motion data in windows.')
    parser.add_argument('--session_id', type=str, required=True, help='The ID of the OpenCap session.')
    parser.add_argument('--trial_name', type=str, required=True, help='The name of the trial to simulate.')
    parser.add_argument('--start_time', type=float, help='Start time of the analysis window.')
    parser.add_argument('--end_time', type=float, help='End time of the analysis window.')
    
    args = parser.parse_args()

    # User Inputs
    session_id = args.session_id
    trial_name = args.trial_name
    session_type = 'overground' # Options are 'overground' and 'treadmill'.
    motion_type = "walking"
    if not 'repetition' in locals():
        repetition = None
    if not 'treadmill_speed' in locals():
        treadmill_speed = 0
    if not 'contact_side' in locals():
        contact_side = 'all'
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
    
    # Use provided start_time and end_time, or determine from mot file if not provided
    if args.start_time is not None:
        start_time = args.start_time
    else:
        start_time, _ = get_mot_time_range(pathTrial)

    if args.end_time is not None:
        end_time = args.end_time
    else:
        _, end_time = get_mot_time_range(pathTrial)

    if start_time is None or end_time is None:
        print("Error: Could not determine time range for analysis. Please provide --start_time and --end_time or ensure the .mot file exists and is valid.")
        return
    
    starts = np.arange(start_time, end_time, 1.0)
    windows = [[s, min(s + 1.0, end_time)] for s in starts]

    if len(windows) > 1:
        last_dur = windows[-1][1] - windows[-1][0]
        if 0 < last_dur < 0.5:
            windows[-2][1] = windows[-1][1]
            windows.pop()

    # Process windows in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(process_single_window, [
            (baseDir, dataFolder, session_id, trial_name, motion_type,
             [win_start, win_end], repetition, treadmill_speed, contact_side,
             solveProblem, analyzeResults, trial_name, i)
            for i, (win_start, win_end) in enumerate(windows)
        ])

    # Separate paths
    grf_file_paths = [r[0] for r in results if r[0] is not None]
    optimaltrajectories_file_paths = [r[1] for r in results if r[1] is not None]

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
                f.write(f"range {df['time'].min()} {df['time'].max()}\n")
                f.write('endheader\n')
                f.write('\t'.join(df.columns) + '\n')
                df.to_csv(f, sep='\t', index=False, header=False)

        write_df_to_mot(full_grf_data, output_grf_file)
        print(f"Full GRF data saved to: {output_grf_file}")
    else:
        print("No GRF data to save.")

    # Concatenate all optimaltrajectories.npy files
    print("Concatenating optimaltrajectories.npy files...")
    full_optimaltrajectories_data = concatenate_npy_files(optimaltrajectories_file_paths)
    print(f"Concatenated optimaltrajectories data contains {len(full_optimaltrajectories_data)} keys.")

    # Optional: Save the full optimaltrajectories data to a new .npy file
    output_npy_file = os.path.join(
        dataFolder, session_id, 'OpenSimData', 'DynamicSimulations', trial_name,
        f'optimaltrajectories_{trial_name}_full.npy')

    if full_optimaltrajectories_data:
        np.save(output_npy_file, full_optimaltrajectories_data)
        print(f"Full optimaltrajectories data saved to: {output_npy_file}")
    else:
        print("No optimaltrajectories data to save.")

    # %% Summary Report
    num_total_windows = len(windows)
    num_successful_windows = len(grf_file_paths) # Since we filter out None
    num_failed_windows = num_total_windows - num_successful_windows

    print("\n--- Processing Summary ---")
    print(f"Total windows attempted: {num_total_windows}")
    print(f"Successfully processed windows: {num_successful_windows}")
    print(f"Failed windows: {num_failed_windows}")
    print("------------------------")

    # %% Plots. (This section will be modified later to handle concatenated data)
    # To compare different cases, add to the cases list, eg cases=['0','1'].
    # plotResultsOpenSimAD(dataFolder, session_id, trial_name, settings, cases=[case])

if __name__ == "__main__":
    main()