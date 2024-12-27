import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Reset matplotlib parameters
inline_rc = dict(mpl.rcParams)

# Define facies colors and labels
facies_colors = ['#F4D03F', '#F5B041', '#DC7633', '#6E2C00',
                 '#1B4F72', '#2E86C1', '#AED6F1', '#A569BD', '#196F3D']

facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
                 'WS', 'D', 'PS', 'BS']

# Create facies color map
facies_color_map = dict(zip(facies_labels, facies_colors))

# Function to map numeric facies to labels
def label_facies(facies_num):
    return facies_labels[facies_num - 1]

# Load your numpy array
facies_data = np.load('/home/dell/disk1/Jinlong/faciesdata/train_labels.npy')

# Convert numpy array to pandas DataFrame
# Assuming your data has these columns - adjust according to your actual data
columns = ['GR', 'ILD_log10', 'PE', 'PHIND', 'NM_M', 'RELPOS', 
           'Facies', 'Formation', 'Depth', 'Cls1', 'Cls7']
df = pd.DataFrame(facies_data, columns=columns)

# Add facies labels column
df['FaciesLabels'] = df['Facies'].apply(label_facies)

# Create the pairplot
plt.figure(figsize=(20, 20))
sns.set_style("whitegrid")
pairplot = sns.pairplot(
    data=df.drop(['Cls1', 'Facies', 'Formation', 'Depth', 'NM_M', 'RELPOS', 'Cls7'], axis=1),
    hue='FaciesLabels',
    palette=facies_color_map,
    hue_order=list(reversed(facies_labels)),
    diag_kind='hist'
)

# Adjust the layout
plt.tight_layout()

# Reset to default matplotlib style
mpl.rcParams.update(inline_rc)
plt.savefig('facies_distribution.png')
# Show the plot
plt.show()