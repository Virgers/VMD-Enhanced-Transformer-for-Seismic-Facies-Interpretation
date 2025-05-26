# import numpy as np
# from tqdm import tqdm
# import concurrent.futures
# from vmdpy import VMD  
# import psutil
# import time

# # f3 vmd shape: (281101, 8, 254) nz vmd shape:  (461380, 8, 1006)
# "Take care of the f file is ok in line 38-41"

# # Dummy values for alpha, tau, K, DC, init, tol
# alpha, tau, K, DC, init, tol = 2000, 0, 8, 0, 1, 1e-7

# # Dummy data as a NumPy array

# u_list = []

# f3facies = np.load('/home/dell/disk1/Jinlong/faciesdata/train_labels.npy')   # (401, 701, 255)
# f3facies = f3facies.reshape(-1, 255)

# NZFacies = np.load("/home/dell/disk1/Jinlong/faciesdata/data_train.npz")     
# NZFacies = NZFacies['data']
# NZFacies = np.swapaxes(NZFacies, 1, 0)
# NZFacies = np.swapaxes(NZFacies, -1, 1)
# NZFacies = NZFacies.reshape(-1, 1006)  # (782, 590, 1006)

# # manupulate with f3 facies
# f = f3facies

# # Function to process each trace
# def process_trace(trace):
#     u, _, _ = VMD(trace, alpha, tau, K, DC, init, tol)
#     return u

# print('Start process vmd!')

# with concurrent.futures.ProcessPoolExecutor() as executor:
#     results = list(tqdm(executor.map(process_trace, f), desc="Processing traces", total=len(f), ncols=100, unit="trace"))

# # Collect results
# u_list = [result for result in results]
# u_array = np.array(u_list)

# print("Processing complete.")

# np.save('full_F3_vmd.npy', u_array)

# print("VMD file saved.")

import numpy as np
from tqdm import tqdm
import concurrent.futures
from vmdpy import VMD  
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Parameters
alpha, tau, K, DC, init, tol = 2000, 0, 8, 0, 1, 1e-7

# Load data
f3facies = np.load('/home/dell/disk1/Jinlong/faciesdata/train_labels.npy')   # (401, 701, 255)
f3facies = f3facies.reshape(-1, 255)

NZFacies = np.load("/home/dell/disk1/Jinlong/faciesdata/data_train.npz")     
NZFacies = NZFacies['data']
NZFacies = np.swapaxes(NZFacies, 1, 0)
NZFacies = np.swapaxes(NZFacies, -1, 1)
NZFacies = NZFacies.reshape(-1, 1006)  # (782, 590, 1006)
NZFacies = NZFacies[:10,]
# Use f3facies data
f = NZFacies
    
# Function to process each trace
def process_trace(trace):
    u, u_hat, omega = VMD(trace, alpha, tau, K, DC, init, tol)
    return u, u_hat, omega

print('Start process vmd!')

# Process only a few traces for visualization
sample_indices = [0, 2, 8]
results = []

for idx in sample_indices:
    if idx < len(f):
        print(f"Processing trace {idx}")
        u, u_hat, omega = process_trace(f[idx])
        results.append((idx, u, u_hat, omega))

print("Processing complete for visualization samples.")

# Visualization part
def visualize_vmd_results(trace_index, u, original_signal):
    # VMD correctly reconstructs by summing the IMFs (but need to verify format)
    reconstructed_signal = np.sum(u, axis=0)
    # Calculate reconstruction error
    error = original_signal - reconstructed_signal
    
    # Create a figure with 5 subfigures
    plt.figure(figsize=(20, 16))
    gs = GridSpec(5, 1, figure=plt.gcf(), hspace=1)
    
    # Plot original signal
    ax1 = plt.subplot(gs[0])
    ax1.plot(original_signal, 'b', linewidth=1)
    ax1.set_title('Original Seismic Signal', fontsize=22)
    ax1.set_xlabel('Sample', fontsize=18)
    ax1.set_ylabel('Amplitude', fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    
    # Plot selected IMFs (first 3 components)
    for i in range(3):
        ax = plt.subplot(gs[i+1])
        ax.plot(u[i], linewidth=1, label=f'IMF {i+1}')
        ax.set_title(f'Decomposed IMF {i+1}', fontsize=22)
        ax.set_xlabel('Sample', fontsize=18)
        ax.set_ylabel('Amplitude', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend(fontsize=18)
    
    # Plot comparison of original and reconstructed
    ax5 = plt.subplot(gs[4])
    ax5.plot(original_signal, 'b', linewidth=1, label='Original')
    ax5.plot(reconstructed_signal, 'r-', linewidth=1, label='Reconstructed')
    # Plot reconstruction error
    ax5.plot(error, 'g:', linewidth=1, label='Error')
    ax5.set_title('Original vs Reconstructed Signal', fontsize=14)
    ax5.set_xlabel('Sample', fontsize=18)
    ax5.set_ylabel('Amplitude', fontsize=18)
    ax5.tick_params(axis='both', which='major', labelsize=14)
    ax5.legend(fontsize=18)
    
    # Add reconstruction quality metrics
    rmse = np.sqrt(np.mean(np.square(error)))
    max_error = np.max(np.abs(error))
    text = f"RMSE: {rmse:.4f}\nMax Error: {max_error:.4f}"
    ax5.text(0.02, 0.02, text, transform=ax5.transAxes, fontsize=8,
             bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f'vmd_visualization_trace_{trace_index}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved for trace {trace_index}")

# Visualize results
for result in results:
    idx, u, _, _ = result
    visualize_vmd_results(idx, u, f[idx])