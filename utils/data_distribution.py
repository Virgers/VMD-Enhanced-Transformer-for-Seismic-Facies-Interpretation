import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

"""

This code is for displaying label distribution

"""

# # facies_volume = np.load('/home/dell/disk1/Jinlong/faciesdata/train_labels.npy')
# facies_volume = np.load('/home/dell/disk1/Jinlong/faciesdata/labels.npy')

# unique_classes, counts = np.unique(facies_volume, return_counts=True)
# total_samples = facies_volume.size

# # Calculate percentages
# percentages = (counts / total_samples) * 100

# # Create a bar plot of the distribution
# plt.figure(figsize=(12, 6))
# plt.bar(unique_classes, percentages)
# plt.title('Distribution of Facies Classes')
# plt.xlabel('Facies Class')
# plt.ylabel('Percentage (%)')

# # Add percentage labels on top of each bar
# for i, v in enumerate(percentages):
#     plt.text(unique_classes[i], v + 0.5, f'{v:.1f}%', ha='center')

# # Add grid for better readability
# plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# # Print the distribution details
# print("Facies Class Distribution:")
# for class_id, count, percentage in zip(unique_classes, counts, percentages):
#     print(f"Class {class_id}: {count:,} samples ({percentage:.1f}%)")
    
# plt.savefig('13.png')
# plt.show()

# Optional: Show distribution along each axis 
# ======== This makes less sense =======
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# # Distribution along inline direction (axis 0)
# inline_dist = np.mean(facies_volume, axis=(1, 2))
# ax1.plot(inline_dist)
# ax1.set_title('Average Distribution Along Inline')
# ax1.set_xlabel('Inline Number')
# ax1.set_ylabel('Average Occurrence')

# # Distribution along crossline direction (axis 1)
# xline_dist = np.mean(facies_volume, axis=(0, 2))
# ax2.plot(xline_dist)
# ax2.set_title('Average Distribution Along Crossline')
# ax2.set_xlabel('Crossline Number')
# ax2.set_ylabel('Average Occurrence')

# # Distribution along depth/time direction (axis 2)
# depth_dist = np.mean(facies_volume, axis=(0, 1))
# ax3.plot(depth_dist)
# ax3.set_title('Average Distribution Along Depth/Time')
# ax3.set_xlabel('Depth/Time Sample')
# ax3.set_ylabel('Average Occurrence')

# plt.tight_layout()
# plt.savefig('11.png')
# plt.show()


# def create_visualization(data_3d, labels):
#     """
#     Create pairplot and correlation matrix visualization from 3D data and labels
    
#     Parameters:
#     data_3d: numpy array of shape (n_samples, n_features, n_dimensions)
#     labels: array of shape (n_samples,) containing class labels
#     """
#     # Flatten the 3D data into 2D
#     n_samples, n_features, n_dims = data_3d.shape
#     flattened_data = data_3d.reshape(n_samples, -1)
    
#     # Create column names for the flattened features
#     column_names = [f'feature_{i}_dim_{j}' for i in range(n_features) 
#                    for j in range(n_dims)]
    
#     # Create DataFrame
#     df = pd.DataFrame(flattened_data, columns=column_names)
#     df['Label'] = labels
    
#     # Create pairplot
#     plt.figure(figsize=(12, 12))
#     pairplot = sns.pairplot(df, hue='Label', diag_kind='kde')
#     plt.title('Pairplot of Features')
    
#     # Create correlation matrix
#     plt.figure(figsize=(10, 8))
#     correlation_matrix = df.drop('Label', axis=1).corr()
#     sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
#     plt.title('Correlation Matrix')
#     plt.savefig('Correlation Matrix.png')
    
#     return pairplot, correlation_matrix

# data_3d = np.random.random((2, 5, 5))

# labels = np.random.randint(0, 2, size=(2, 5, 5))

# pairplot, corr_matrix = create_visualization(data_3d, labels)

# Randomly generate labels for the data
# Assuming binary labels for simplicity (0 or 1)
# Example usage:
# Assuming you have your 3D numpy array as data_3d and labels

# data_3d = np.load('/home/dell/disk1/Jinlong/faciesdata/train_seismic.npy')
# labels = np.load('/home/dell/disk1/Jinlong/faciesdata/train_labels.npy')
# data_3d = data_3d[::100,::100,::10]
# labels = labels[::100,::100,::10]


def create_visualization(data_3d, labels):
    """
    Create pairplot and correlation matrix visualization from 3D data and labels
    
    Parameters:
    data_3d: numpy array of shape (n_samples, n_features, n_dimensions)
    labels: array of shape (n_samples,) containing class labels
    """
    # Flatten the 3D data into 2D
    n_samples, n_features, n_dims = data_3d.shape
    flattened_data = data_3d.reshape(n_samples, -1)
    
    # Flatten the labels to match the number of samples
    flattened_labels = labels.reshape(-1)
    
    # Ensure labels and data are aligned
    assert flattened_data.shape[0] == flattened_labels.shape[0], "Mismatch between data and labels size."
    
    # Create column names for the flattened features
    column_names = [f'feature_{i}_dim_{j}' for i in range(n_features) 
                   for j in range(n_dims)]
    
    # Create DataFrame
    df = pd.DataFrame(flattened_data, columns=column_names)
    df['Label'] = flattened_labels
    
    # Create pairplot
    plt.figure(figsize=(12, 12))
    pairplot = sns.pairplot(df, hue='Label', diag_kind='kde')
    plt.title('Pairplot of Features')
    plt.savefig('Pairplot of Features.png')
    plt.show()
    
    # Create correlation matrix
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.drop('Label', axis=1).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.savefig('Correlation Matrix.png')
    plt.show()
    
    return pairplot, correlation_matrix

# Test the function with some random data and labels
data_3d = np.random.random((10, 5, 5))  # 10 samples, 5 features, 5 dimensions
labels = np.random.randint(0, 2, size=(10,))  # 10 labels for the 10 samples

pairplot, corr_matrix = create_visualization(data_3d, labels)
