import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

"""

This code is for displaying label distribution

"""
def select_random_traces(seismic_volume, n_traces=5, seed=42):
    """
    Randomly select traces from a 3D seismic volume
    
    Parameters:
    seismic_volume (numpy.ndarray): 3D array with shape (time/depth, inline, crossline)
    n_traces (int): Number of traces to select
    seed (int): Random seed for reproducibility
    
    Returns:
    tuple: Selected traces and their positions (inline, crossline)
    """
    np.random.seed(seed)
    
    # Get volume dimensions
    n_inline, n_crossline, n_samples = seismic_volume.shape
    
    # Generate random positions
    total_traces = n_inline * n_crossline
    flat_indices = np.random.choice(total_traces, n_traces, replace=False)
    
    # Convert to inline/crossline positions
    inline_pos = flat_indices // n_crossline
    crossline_pos = flat_indices % n_crossline
    
    # Extract traces
    selected_traces = np.array([
        seismic_volume[:, il, xl] 
        for il, xl in zip(inline_pos, crossline_pos)
    ]).T  # Shape: (n_samples, n_traces)
    
    return selected_traces, list(zip(inline_pos, crossline_pos))

seismic_volume = np.load('/home/dell/disk1/Jinlong/faciesdata/train_seismic.npy')
n_traces = 5
seed = 42
traces, positions = select_random_traces(seismic_volume, n_traces, seed)

def create_visualization(data, features, label_column):
    """
    Create pairplot and correlation matrix for given features and label
    
    Parameters:
    data (pandas.DataFrame): Input dataframe
    features (list): List of feature column names
    label_column (str): Name of the label column
    """
    # Select relevant columns
    plot_data = data[features + [label_column]].copy()
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Create pairplot
    sns.set_style("whitegrid")
    pairplot = sns.pairplot(plot_data, 
                           hue=label_column,
                           diag_kind="kde",
                           plot_kws={'alpha': 0.6},
                           diag_kws={'alpha': 0.6})
    
    # Calculate correlation matrix
    correlation_matrix = plot_data.corr()
    
    # Create heatmap
    sns.heatmap(correlation_matrix,
                annot=True,
                cmap='coolwarm',
                vmin=-1,
                vmax=1,
                center=0,
                ax=ax2)
    
    ax2.set_title('Correlation Matrix')
    
    # Adjust layout
    plt.tight_layout()
    
    return pairplot, fig

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    data_3d = np.load('/home/dell/disk1/Jinlong/faciesdata/train_seismic.npy')
    labels = np.load('/home/dell/disk1/Jinlong/faciesdata/train_labels.npy')
    data_3d = data_3d[::100,::100,::10]
    labels = labels[::100,::100,::10]
    
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'label': np.random.random_integers(0,7, n_samples)
    })
    
    features = ['feature1', 'feature2']
    pairplot, correlation_fig = create_visualization(data, features, 'label')
    
    # Show plots
    plt.savefig('test.png')
    plt.show()