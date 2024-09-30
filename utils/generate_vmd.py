import numpy as np
from tqdm import tqdm
import concurrent.futures
from vmdpy import VMD  
import psutil
import time

# f3 vmd shape: (281101, 8, 254) nz vmd shape:  (461380, 8, 1006)
"Take care of the f file is ok in line 38-41"

# Dummy values for alpha, tau, K, DC, init, tol
alpha, tau, K, DC, init, tol = 2000, 0, 8, 0, 1, 1e-7

# Dummy data as a NumPy array

u_list = []

f3facies = np.load('/home/dell/disk1/Jinlong/faciesdata/train_labels.npy')   # (401, 701, 255)
f3facies = f3facies.reshape(-1, 255)

NZFacies = np.load("/home/dell/disk1/Jinlong/faciesdata/data_train.npz")     
NZFacies = NZFacies['data']
NZFacies = np.swapaxes(NZFacies, 1, 0)
NZFacies = np.swapaxes(NZFacies, -1, 1)
NZFacies = NZFacies.reshape(-1, 1006)  # (782, 590, 1006)

# manupulate with f3 facies
f = f3facies

# Function to process each trace
def process_trace(trace):
    u, _, _ = VMD(trace, alpha, tau, K, DC, init, tol)
    return u

print('Start process vmd!')

with concurrent.futures.ProcessPoolExecutor() as executor:
    results = list(tqdm(executor.map(process_trace, f), desc="Processing traces", total=len(f), ncols=100, unit="trace"))

# Collect results
u_list = [result for result in results]
u_array = np.array(u_list)

print("Processing complete.")

np.save('full_F3_vmd.npy', u_array)

print("VMD file saved.")