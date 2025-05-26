import os
import numpy as np
import segyio
import multiprocessing as mp
from tqdm import tqdm

class SegyConverter:
    def __init__(self, input_file, output_file):
        """
        Initialize the converter with input and output file paths
        
        Args:
            input_file (str): Path to input SEG-Y file
            output_file (str): Path to output NPY file
        """
        self.input_file = input_file
        self.output_file = output_file
        
    def convert_small_file(self):
        """
        Convert small SEG-Y files to NPY format using direct memory loading
        """
        try:
            with segyio.open(self.input_file, 'r', ignore_geometry=True) as segy_file:
               
                n_traces = len(segy_file.trace)
                n_samples = len(segy_file.samples)
                
                # Pre-allocate the numpy array
                data = np.zeros((n_traces, n_samples), dtype='float32')
                
              
                for i in tqdm(range(n_traces), desc="Converting traces"):
                    data[i] = segy_file.trace[i]
                    
               
                np.save(self.output_file, data)
                print(f"Successfully converted to {self.output_file}")
                print(f"Output shape: {data.shape}")
                
        except Exception as e:
            print(f"Error converting file: {str(e)}")
            
    def convert_large_file(self, chunk_size=1000):
        """
        Convert large SEG-Y files to NPY format using memory mapping
        
        Args:
            chunk_size (int): Number of traces to process at once
        """
        try:
            with segyio.open(self.input_file, 'r', ignore_geometry=True) as segy_file:
                # Get dimensions
                n_traces = len(segy_file.trace)
                n_samples = len(segy_file.samples)
                
                # Create memory-mapped file
                mmap_data = np.memmap(self.output_file, 
                                    dtype='float32',
                                    mode='w+',
                                    shape=(n_traces, n_samples))
                
                # Process in chunks
                for chunk_start in tqdm(range(0, n_traces, chunk_size),
                                      desc="Converting chunks"):
                    chunk_end = min(chunk_start + chunk_size, n_traces)
                    
                    # Read chunk of traces
                    chunk_data = np.array([segy_file.trace[i] 
                                         for i in range(chunk_start, chunk_end)],
                                        dtype='float32')
                    
                    # Write to memory-mapped file
                    mmap_data[chunk_start:chunk_end] = chunk_data
                    
                # Flush changes to disk
                mmap_data.flush()
                
                print(f"Successfully converted to {self.output_file}")
                print(f"Output shape: ({n_traces}, {n_samples})")
                
        except Exception as e:
            print(f"Error converting file: {str(e)}")
            
    def parallel_convert(self, num_processes=None):
        """
        Convert SEG-Y file using parallel processing
        
        Args:
            num_processes (int): Number of processes to use. If None, uses 75% of CPU cores
        """
        if num_processes is None:
            num_processes = max(1, int(mp.cpu_count() * 0.75))
            
        try:
            with segyio.open(self.input_file, 'r', ignore_geometry=True) as segy_file:
                n_traces = len(segy_file.trace)
                n_samples = len(segy_file.samples)
                
                # Calculate chunks for each process
                chunk_size = n_traces // num_processes
                chunks = []
                
                for i in range(num_processes):
                    start_idx = i * chunk_size
                    end_idx = start_idx + chunk_size if i < num_processes-1 else n_traces
                    chunk_file = f'temp_chunk_{i}.npy'
                    chunks.append((start_idx, end_idx, chunk_file))
                
                # Process chunks in parallel
                with mp.Pool(processes=num_processes) as pool:
                    args = [(self.input_file, start, end, tmp_file) 
                           for start, end, tmp_file in chunks]
                    
                    chunk_files = list(tqdm(pool.imap(self._process_chunk, args),
                                          total=len(chunks),
                                          desc="Processing chunks"))
                
                # Combine chunks
                print("Combining chunks...")
                final_data = np.zeros((n_traces, n_samples), dtype='float32')
                
                for i, (_, _, chunk_file) in enumerate(chunks):
                    start_idx = chunks[i][0]
                    end_idx = chunks[i][1]
                    chunk_data = np.load(chunk_file)
                    final_data[start_idx:end_idx] = chunk_data
                    
                    # Clean up temporary file
                    os.remove(chunk_file)
                
                # Save final combined array
                np.save(self.output_file, final_data)
                print(f"Successfully converted to {self.output_file}")
                print(f"Output shape: {final_data.shape}")
                
        except Exception as e:
            print(f"Error in parallel conversion: {str(e)}")
    
    @staticmethod
    def _process_chunk(args):
        """
        Process a chunk of traces (used by parallel_convert)
        """
        input_file, start_idx, end_idx, chunk_file = args
        
        with segyio.open(input_file, 'r', ignore_geometry=True) as segy_file:
            chunk_data = np.array([segy_file.trace[i] 
                                 for i in range(start_idx, end_idx)],
                                dtype='float32')
            np.save(chunk_file, chunk_data)
            
        return chunk_file

if __name__ == "__main__":
    # File paths
    sgy_file = "/home/dell/disk1/Jinlong/faciesdata/SEG2020/TestData_Image2.segy"
    npy_file = "TestData_Image2.npy"
    
    converter = SegyConverter(sgy_file, npy_file)
    
    file_size = os.path.getsize(sgy_file)
    
    if file_size < 1e9:  # Less than 1GB
        converter.convert_small_file()
    elif file_size < 10e9:  # Less than 10GB
        converter.convert_large_file(chunk_size=1000)
    else:  # Larger files
        converter.parallel_convert()