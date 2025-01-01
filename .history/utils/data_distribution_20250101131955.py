import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

"""

This code is for displaying label distribution

"""
def select_random_traces(seismic_volume, label_volume, n_traces=5, seed=42):
    """
    Randomly select traces from a 3D seismic volume
    
    Parameters:
    seismic_volume (numpy.ndarray): list of 3D array with shape (time/depth, inline, crossline)
    label_volume (numpy.ndarray): 3D array with shape (time/depth, inline, crossline)
    n_traces (int): Number of traces to select
    seed (int): Random seed for reproducibility
    
    Returns:
    tuple: Selected traces list, labels, and their positions (inline, crossline)
    """
    np.random.seed(seed)
    selected_traces_volume = []
    # Get volume dimensions
    n_inline, n_crossline, n_samples = seismic_volume[0].shape
    
    # Generate random positions
    total_traces = n_inline * n_crossline
    flat_indices = np.random.choice(total_traces, n_traces, replace=False)
    
    # Convert to inline/crossline positions
    inline_pos = flat_indices // n_crossline
    crossline_pos = flat_indices % n_crossline
    
    # Extract traces
    for i in range(len(seismic_volume)):
        selected_traces = np.array([
            seismic_volume[i][il, xl, :] 
            for il, xl in zip(inline_pos, crossline_pos)
        ]).T  # Shape: (n_samples, n_traces)
        
        selected_traces_volume.append(selected_traces)
        
    selected_label_traces = np.array([
        label_volume[il, xl, :] 
        for il, xl in zip(inline_pos, crossline_pos)
    ]).T 
    
    return selected_traces_volume, selected_label_traces, list(zip(inline_pos, crossline_pos))


def prepare_trace_data(traces_volume, labels, positions, attr_name):
    """
    Prepare data for each trace position
    
    Parameters:
    traces_volume (numpy.ndarray): Selected traces for different time samples
    labels (numpy.ndarray): Labels for each trace
    positions (list): List of (inline, crossline) positions
    
    Returns:
    list: List of DataFrames, one for each trace position
    """
    df_list = []
    
    
    for i, (il, xl) in enumerate(positions):
        
        trace_dict = {
            f'{attr_name[j]}_Inline_{il}_Crossline_{xl}':  traces_volume[j][:, i] 
            for j in range(len(traces_volume))
        }
        
        df = pd.DataFrame(trace_dict)
        df['label'] = labels[:, i]  
        
        df_list.append(df)
    
    return df_list


def create_visualization(data, file_name):
    """
    Create pairplot and correlation matrix for given features and label
    
    Parameters:
    data (pandas.DataFrame): Input dataframe
    features (list): List of feature column names
    label_column (str): Name of the label column
    """
    
    plot_data = data
    label_column = 'label'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    sns.set_style("whitegrid")
    pairplot = sns.pairplot(plot_data, 
                           hue=label_column,
                           diag_kind="kde",
                           plot_kws={'alpha': 0.6},
                           diag_kws={'alpha': 0.6})
    
    correlation_matrix = plot_data.corr()
    
    sns.heatmap(correlation_matrix,
                annot=True,
                cmap='coolwarm',
                vmin=-1,
                vmax=1,
                center=0,
                ax=ax2)
    
    ax2.set_title('Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(f'{file_name}_correlation_matrix.png')
    # plt.show()
    
    return pairplot, fig


if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    attr_name = ['seismic', 'dip', 'phase', 'rms']
    
    seismic_volume_1 = np.load('/home/dell/disk1/Jinlong/faciesdata/train_seismic.npy')
    seismic_volume_3 = seismic_volume_2 = seismic_volume_1
    
    seismic_labels = np.load('/home/dell/disk1/Jinlong/faciesdata/train_labels.npy')
    
    n_traces = 5
    seed = 42
    seismic_volume = [seismic_volume_1,seismic_volume_2, seismic_volume_3]
    selected_traces_volume, selected_labels, positions = select_random_traces(seismic_volume, seismic_labels, n_traces, seed)
    
    df_list = prepare_trace_data(selected_traces_volume, selected_labels, positions, attr_name)
    
    for i, (il, xl) in enumerate(positions):
        file_name = f'Inline_{il}_Crossline_{xl}'
        pairplot, correlation_fig = create_visualization(df_list[i], file_name)
        