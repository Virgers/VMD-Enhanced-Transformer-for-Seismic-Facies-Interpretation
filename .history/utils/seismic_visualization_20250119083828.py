import os
import numpy as np
import segyio
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# SEG2020 Train data and labels 590, 782, 1006
# Test data1 251, 782, 1006 test2 841, 334, 1006

class SeismicDataAnalyzer:
    def __init__(self, output_dir='seismic_output'):
        """
        Initialize the analyzer with input and output paths
        
        Args:
            output_dir (str): Directory to save outputs (will be created if doesn't exist)
        """
        self.data_paths = {
            'train_image': "/home/dell/disk1/Jinlong/faciesdata/SEG2020/TrainingData_Image.segy",
            'train_labels': "/home/dell/disk1/Jinlong/faciesdata/SEG2020/TrainingData_Labels.segy",
            'test1': "/home/dell/disk1/Jinlong/faciesdata/SEG2020/TestData_Image1.segy",
            'test2': "/home/dell/disk1/Jinlong/faciesdata/SEG2020/TestData_Image2.segy"
        }
        self.data_shapes = {}
        self.data_3d = {}
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
            
    def save_visualization(self, fig, filename):
        """Save visualization figure"""
        output_path = os.path.join(self.output_dir, filename)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")

    def read_segy_shape(self, file_path):
        """Read SEG-Y file and determine its shape"""
        with segyio.open(file_path, 'r', ignore_geometry=True) as segy:
            n_traces = len(segy.trace)
            n_samples = len(segy.samples)
            print(f"File: {os.path.basename(file_path)}")
            print(f"Number of traces: {n_traces}")
            print(f"Samples per trace: {n_samples}")
            print(f"2D Shape: ({n_traces}, {n_samples})\n")
            return n_traces, n_samples

    def get_3d_dimensions(self, segy_file):
        """Determine inline, crossline, and time dimensions"""
        with segyio.open(segy_file, 'r') as segy:
            # Get unique inline and crossline numbers
            inlines = segy.attributes(segyio.TraceField.INLINE_3D)[:]
            crosslines = segy.attributes(segyio.TraceField.CROSSLINE_3D)[:]
            
            n_ilines = len(np.unique(inlines))
            n_xlines = len(np.unique(crosslines))
            n_samples = len(segy.samples)
            
            return n_ilines, n_xlines, n_samples

    def load_segy_to_3d(self, file_path):
        """Load SEG-Y file into 3D numpy array"""
        with segyio.open(file_path, 'r') as segy:
            n_ilines, n_xlines, n_samples = self.get_3d_dimensions(file_path)
            data_3d = np.zeros((n_ilines, n_xlines, n_samples), dtype=np.float32)
            
            # Read all traces
            traces = segy.trace.raw[:]  # Read all traces at once
            
            # Reshape into 3D volume
            data_3d = traces.reshape((n_ilines, n_xlines, n_samples))
            return data_3d

    def analyze_all_files(self):
        """Analyze shapes of all SEG-Y files"""
        for name, path in self.data_paths.items():
            self.data_shapes[name] = self.read_segy_shape(path)
            try:
                self.data_3d[name] = self.load_segy_to_3d(path)
                print(f"3D shape for {name}: {self.data_3d[name].shape}")
            except Exception as e:
                print(f"Could not load {name} as 3D volume: {str(e)}")

    def visualize_section(self, data, section_type, index, title, cmap='seismic'):
        """Visualize a section of 3D seismic data"""
        plt.figure(figsize=(12, 8))
        
        if section_type == 'inline':
            section = data[index, :, :]
            xlabel, ylabel = 'Crossline', 'Time'
        elif section_type == 'crossline':
            section = data[:, index, :]
            xlabel, ylabel = 'Inline', 'Time'
        else:  # time slice
            section = data[:, :, index]
            xlabel, ylabel = 'Inline', 'Crossline'

        ax = plt.gca()
        im = ax.imshow(section, cmap=cmap, aspect='auto')
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        plt.title(f"{title} - {section_type.capitalize()} Section {index}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        return plt.gcf()

    def visualize_all_sections(self, data_key, inline_idx=None, xline_idx=None, time_idx=None):
        """Visualize inline, crossline, and time sections for a dataset"""
        if data_key not in self.data_3d:
            print(f"No 3D data loaded for {data_key}")
            return

        data = self.data_3d[data_key]
        n_ilines, n_xlines, n_samples = data.shape

        # Use middle indices if not specified
        inline_idx = inline_idx if inline_idx is not None else n_ilines // 2
        xline_idx = xline_idx if xline_idx is not None else n_xlines // 2
        time_idx = time_idx if time_idx is not None else n_samples // 2

        # Create three subplots
        fig = plt.figure(figsize=(20, 6))
        
        # Inline section
        plt.subplot(131)
        plt.imshow(data[inline_idx, :, :], aspect='auto', cmap='seismic')
        plt.title(f'Inline {inline_idx}')
        plt.xlabel('Crossline')
        plt.ylabel('Time')
        
        # Crossline section
        plt.subplot(132)
        plt.imshow(data[:, xline_idx, :], aspect='auto', cmap='seismic')
        plt.title(f'Crossline {xline_idx}')
        plt.xlabel('Inline')
        plt.ylabel('Time')
        
        # Time slice
        plt.subplot(133)
        plt.imshow(data[:, :, time_idx], aspect='auto', cmap='seismic')
        plt.title(f'Time Slice {time_idx}')
        plt.xlabel('Inline')
        plt.ylabel('Crossline')
        
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    # Create analyzer with output directory
    analyzer = SeismicDataAnalyzer(output_dir='seismic_output')
    
    analyzer.analyze_all_files()
    
    # Visualize sections for training data
    if 'train_image' in analyzer.data_3d:
        fig = analyzer.visualize_all_sections('train_image')
        plt.show()
        
        # Save all data and visualizations
        analyzer.save_visualization(fig, 'seismic_sections_train.png')
        
        # Generate and save visualizations for all datasets
        for data_key in analyzer.data_3d:
            fig = analyzer.visualize_all_sections(data_key)
            analyzer.save_visualization(fig, f'seismic_sections_{data_key}.png')
            plt.close(fig)  # Close figure to free memory