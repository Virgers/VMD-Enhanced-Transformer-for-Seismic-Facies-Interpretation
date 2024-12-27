import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Reset matplotlib parameters
inline_rc = dict(mpl.rcParams)

# Define facies colors and labels
facies_colors = ['#F4D03F', '#F5B041', '#DC7633', '#6E2C00',
                 '#1B4F72', '#2E86C1', '#AED6F1', '#A569BD', '#196F3D']

facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
                 'WS', 'D', 'PS', 'BS']

# Create a custom colormap from the facies colors
facies_cmap = mpl.colors.ListedColormap(facies_colors)

# Load the 3D volume
facies_volume = np.load('/home/dell/disk1/Jinlong/faciesdata/train_labels.npy')

# Create subplots for different views
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# Plot inline view (XZ plane at middle Y)
mid_inline = facies_volume[:, facies_volume.shape[1]//2, :]
im1 = ax1.imshow(mid_inline.T, cmap=facies_cmap, aspect='auto')
ax1.set_title('Inline View (Middle)')
ax1.set_xlabel('Inline')
ax1.set_ylabel('Time/Depth')

# Plot crossline view (YZ plane at middle X)
mid_xline = facies_volume[facies_volume.shape[0]//2, :, :]
im2 = ax2.imshow(mid_xline.T, cmap=facies_cmap, aspect='auto')
ax2.set_title('Crossline View (Middle)')
ax2.set_xlabel('Crossline')
ax2.set_ylabel('Time/Depth')

# Plot time slice (XY plane at middle Z)
time_slice = facies_volume[:, :, facies_volume.shape[2]//2]
im3 = ax3.imshow(time_slice, cmap=facies_cmap, aspect='auto')
ax3.set_title('Time Slice (Middle)')
ax3.set_xlabel('Inline')
ax3.set_ylabel('Crossline')

# Add colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = plt.colorbar(im1, cax=cbar_ax)
cbar.set_ticks(np.arange(len(facies_labels)) + 0.5)
cbar.set_ticklabels(facies_labels)

# Adjust layout
plt.tight_layout()

# Reset to default matplotlib style
mpl.rcParams.update(inline_rc)

plt.savefig('facies_distribution.png')
# Show the plot
plt.show()

# Print volume information
print(f"Volume shape: {facies_volume.shape}")
print(f"Unique facies values: {np.unique(facies_volume)}")