import os
import sys

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


baseDir = os.getcwd()
opensimADDir = os.path.join(baseDir, 'UtilsDynamicSimulations', 'OpenSimAD')
sys.path.append(baseDir)
sys.path.append(opensimADDir)

session_id = "OpenCapData_ab7eb7cf-817d-4035-a30b-ee68773906cb"
trial_name = 'Suhasno_2'

dataFolder = baseDir

sessionFolder =  os.path.join(dataFolder, session_id)
pathTrial = os.path.join(sessionFolder, 'OpenSimData', 'Kinematics', trial_name + '.mot') 
start_time, end_time = get_mot_time_range(pathTrial)
print(start_time, end_time)