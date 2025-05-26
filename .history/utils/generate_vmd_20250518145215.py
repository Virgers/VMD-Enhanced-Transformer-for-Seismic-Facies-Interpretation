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
def visualize_vmd_results(trace_index, u, original_signal, omega):
    # VMD correctly reconstructs by summing the IMFs (but need to verify format)
    reconstructed_signal = np.sum(u, axis=0)
    # Calculate reconstruction error
    error = original_signal - reconstructed_signal
    
    # Time domain for x-axis
    time = np.arange(len(original_signal))
    
    # Create a figure with 5 subfigures
    plt.figure(figsize=(20, 16))
    gs = GridSpec(5, 1, figure=plt.gcf(), hspace=1)
    
    # Plot original signal
    ax1 = plt.subplot(gs[0])
    ax1.plot(time, original_signal, 'b', linewidth=1)
    ax1.set_title('Original Seismic Signal', fontsize=22)
    ax1.set_ylabel('Amplitude', fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    
    # Plot selected IMFs (first 3 components)
    for i in range(3):
        ax = plt.subplot(gs[i+1])
        ax.plot(time, u[i], linewidth=1, label=f'IMF {i+1}')
        
        # Get center frequency for this IMF and convert to more readable form
        center_freq = omega[i, -1]  # Use the final iteration's center frequency
        
        # Add center frequency to the title
        ax.set_title(f'IMF {i+1}: Center Frequency = {center_freq:.4f} rad/sample', fontsize=22)
        ax.set_ylabel('Amplitude', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
    
    # Plot comparison of original and reconstructed
    ax5 = plt.subplot(gs[4])
    ax5.plot(time, original_signal, 'b', linewidth=1, label='Original')
    ax5.plot(time, reconstructed_signal, 'r-', linewidth=1, label='Reconstructed')
    # Plot reconstruction error
    ax5.plot(time, error, 'g:', linewidth=1, label='Error')
    ax5.set_title('Original vs Reconstructed Signal', fontsize=22)
    ax5.set_xlabel('Sample Index', fontsize=18)
    ax5.set_ylabel('Amplitude', fontsize=18)
    ax5.tick_params(axis='both', which='major', labelsize=16)
    ax5.legend(fontsize=16)
    
    # Add a secondary axis to show frequency distribution
    ax_freq = plt.figure(figsize=(8, 6))
    plt.stem(np.arange(len(omega[:, -1])), omega[:, -1], 'b-', 'bo', basefmt=" ")
    plt.title('Center Frequencies of All IMFs', fontsize=22)
    plt.xlabel('IMF Index', fontsize=18)
    plt.ylabel('Center Frequency (rad/sample)', fontsize=18)
    plt.xticks(np.arange(len(omega[:, -1])), fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'vmd_frequencies_trace_{trace_index}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(plt.gcf().number)  # Return to the main figure
    plt.tight_layout()
    plt.savefig(f'vmd_visualization_trace_{trace_index}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved for trace {trace_index}")

# Visualize results
for result in results:
    idx, u, _, omega = result
    visualize_vmd_results(idx, u, f[idx], omega)