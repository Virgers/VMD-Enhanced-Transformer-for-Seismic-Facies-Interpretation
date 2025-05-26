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
import matplotlib.pyplot as plt
import os

# 加载VMD结果和原始数据
vmd_results = np.load('full_F3_vmd.npy')
f3facies = np.load('/home/dell/disk1/Jinlong/faciesdata/train_labels.npy')
f3facies = f3facies.reshape(-1, 255)

# 选择一个示例信号进行可视化
trace_idx = 100  # 可以选择任意索引
original_signal = f3facies[trace_idx]
imfs = vmd_results[trace_idx]  # 分解得到的IMF

# 通过求和所有IMF得到重构信号
reconstructed_signal = np.sum(imfs, axis=0)

# 检查并调整信号长度（如果需要）
if len(original_signal) != len(reconstructed_signal):
    min_length = min(len(original_signal), len(reconstructed_signal))
    original_signal = original_signal[:min_length]
    reconstructed_signal = reconstructed_signal[:min_length]

# 创建包含5个子图的图像
fig, axs = plt.subplots(5, 1, figsize=(12, 15), sharex=True)

# 1. 原始信号
axs[0].plot(original_signal, 'k', linewidth=2)
axs[0].set_title('Original Signal', fontsize=18)
axs[0].set_ylabel('Amplitude', fontsize=18)
axs[0].tick_params(axis='both', labelsize=16)

# 定义IMF组 - 低、中、高频率
imf_groups = [[0, 1, 2], [3, 4, 5], [6, 7]]
group_names = ['Low Frequency IMFs (1-3)', 'Mid Frequency IMFs (4-6)', 'High Frequency IMFs (7-8)']

# 2-4. 绘制IMF组
for i, (group, name) in enumerate(zip(imf_groups, group_names)):
    for idx in group:
        axs[i+1].plot(imfs[idx], linewidth=2, label=f'IMF {idx+1}')
    axs[i+1].set_title(name, fontsize=18)
    axs[i+1].set_ylabel('Amplitude', fontsize=18)
    axs[i+1].tick_params(axis='both', labelsize=16)
    axs[i+1].legend(fontsize=16)

# 5. 原始信号与重构信号的对比
axs[4].plot(original_signal, 'k', linewidth=2, label='Original')
axs[4].plot(reconstructed_signal, 'r--', linewidth=2, label='Reconstructed')
axs[4].set_title('Original vs. Reconstructed Signal', fontsize=18)
axs[4].set_xlabel('Sample', fontsize=18)
axs[4].set_ylabel('Amplitude', fontsize=18)
axs[4].tick_params(axis='both', labelsize=16)
axs[4].legend(fontsize=18)

# 调整布局并保存
plt.tight_layout()
plt.savefig('vmd_visualization.png', dpi=300, bbox_inches='tight')
print("图像已保存为 'vmd_visualization.png'")
plt.show()