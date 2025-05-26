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
f3facies = np.load('/home/dell/disk1/Jinlong/faciesdata/train_labels.npy')
f3facies = f3facies.reshape(-1, 255)

NZFacies = np.load("/home/dell/disk1/Jinlong/faciesdata/data_train.npz")     
NZFacies = NZFacies['data']
NZFacies = np.swapaxes(NZFacies, 1, 0)
NZFacies = np.swapaxes(NZFacies, -1, 1)
NZFacies = NZFacies.reshape(-1, 1006)
NZFacies = NZFacies[:10,]

# Use NZFacies data
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

# Visualization part - similar to the provided example
def visualize_vmd_results(trace_index, u, u_hat, omega, original_signal):
    # VMD correctly reconstructs by summing the IMFs
    reconstructed_signal = np.sum(u, axis=0)
    
    # Create a figure with 5 subfigures
    fig, axs = plt.subplots(5, 1, figsize=(12, 10), gridspec_kw={'hspace': 0.4})
    
    # Sample length for x-axis
    n_samples = len(original_signal)
    time_axis = np.arange(n_samples)
    
    # Plot original signal
    axs[0].plot(time_axis, original_signal, 'blue', linewidth=1.5)
    axs[0].set_title('Original Signal', fontsize=14)
    axs[0].set_ylabel('Amplitude', fontsize=12)
    axs[0].set_xlim(0, n_samples)
    
    # Convert omega to Hz for display - handle array case
    fs = 1.0  # Normalized sampling frequency
    
    # Handle different shapes of omega
    if np.isscalar(omega[0]):
        # If omega is already scalar per mode
        center_freqs_hz = omega * fs / (2*np.pi)
    else:
        # If omega is array per mode, take the final value (most converged)
        center_freqs_hz = np.array([om[-1] if len(np.atleast_1d(om)) > 0 else om for om in omega]) * fs / (2*np.pi)
    
    # Select IMFs to highlight (for this example, choose first two)
    selected_imfs = [0, 1, 2]
    
    # Plot selected IMFs (first 3 components)
    for i in range(3):
        # Get the center frequency as a scalar
        if isinstance(center_freqs_hz[i], (np.ndarray, list)):
            freq_value = center_freqs_hz[i][-1] if len(center_freqs_hz[i]) > 0 else 0
        else:
            freq_value = center_freqs_hz[i]
            
        if i in selected_imfs:
            axs[i+1].plot(time_axis, u[i], 'red', linewidth=1.5)
            axs[i+1].set_title(f'IMF {i+1} (Center Freq: {freq_value:.2f} Hz) - Selected', fontsize=14)
        else:
            axs[i+1].plot(time_axis, u[i], 'blue', linewidth=1.5)
            axs[i+1].set_title(f'IMF {i+1} (Center Freq: {freq_value:.2f} Hz)', fontsize=14)
        
        axs[i+1].set_ylabel('Amplitude', fontsize=12)
        axs[i+1].set_xlim(0, n_samples)
    
    # Plot comparison of original and reconstructed
    axs[4].plot(time_axis, original_signal, 'blue', linewidth=1, label='Original')
    axs[4].plot(time_axis, reconstructed_signal, 'red', linewidth=1, label='Reconstructed')
    axs[4].set_title('Original vs Reconstructed Signal', fontsize=14)
    axs[4].set_xlabel('Time (s)', fontsize=12)
    axs[4].set_ylabel('Amplitude', fontsize=12)
    axs[4].set_xlim(0, n_samples)
    axs[4].legend(fontsize=10)
    
    # Adjust y-limits for better visual comparison if needed
    for ax in axs:
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'vmd_visualization_trace_{trace_index}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved for trace {trace_index}")

# Visualize results
for result in results:
    idx, u, u_hat, omega = result
    visualize_vmd_results(idx, u, u_hat, omega, f[idx])