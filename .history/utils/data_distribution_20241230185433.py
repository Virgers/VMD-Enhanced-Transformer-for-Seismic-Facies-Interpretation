import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the 3D volume
facies_volume = np.load('/home/dell/disk1/Jinlong/faciesdata/train_labels.npy')

# Calculate unique values and their counts
unique_classes, counts = np.unique(facies_volume, return_counts=True)
total_samples = facies_volume.size

# Calculate percentages
percentages = (counts / total_samples) * 100

# Create a bar plot of the distribution
plt.figure(figsize=(12, 6))
plt.bar(unique_classes, percentages)
plt.title('Distribution of Facies Classes')
plt.xlabel('Facies Class')
plt.ylabel('Percentage (%)')

# Add percentage labels on top of each bar
for i, v in enumerate(percentages):
    plt.text(unique_classes[i], v + 0.5, f'{v:.1f}%', ha='center')

# Add grid for better readability
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Print the distribution details
print("Facies Class Distribution:")
for class_id, count, percentage in zip(unique_classes, counts, percentages):
    print(f"Class {class_id}: {count:,} samples ({percentage:.1f}%)")
    
plt.savefig('12.png')
plt.show()

# Optional: Show distribution along each axis
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Distribution along inline direction (axis 0)
inline_dist = np.mean(facies_volume, axis=(1, 2))
ax1.plot(inline_dist)
ax1.set_title('Average Distribution Along Inline')
ax1.set_xlabel('Inline Number')
ax1.set_ylabel('Average Occurrence')

# Distribution along crossline direction (axis 1)
xline_dist = np.mean(facies_volume, axis=(0, 2))
ax2.plot(xline_dist)
ax2.set_title('Average Distribution Along Crossline')
ax2.set_xlabel('Crossline Number')
ax2.set_ylabel('Average Occurrence')

# Distribution along depth/time direction (axis 2)
depth_dist = np.mean(facies_volume, axis=(0, 1))
ax3.plot(depth_dist)
ax3.set_title('Average Distribution Along Depth/Time')
ax3.set_xlabel('Depth/Time Sample')
ax3.set_ylabel('Average Occurrence')

plt.tight_layout()
plt.savefig('11.png')
plt.show()