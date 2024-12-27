inline_rc = dict(mpl.rcParams)

import seaborn as sns
import numpy as np


facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00',
       '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']

facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
                 'WS', 'D','PS', 'BS']
#facies_color_map is a dictionary that maps facies labels
#to their respective colors
facies_color_map = {}
for ind, label in enumerate(facies_labels):
    facies_color_map[label] = facies_colors[ind]

def label_facies(row, labels):
    return labels[ row['Facies'] -1]
    
facies_label = np.load('/home/dell/disk1/Jinlong/faciesdata/train_label.npy')

facies_label.loc[:,'FaciesLabels'] = facies_label.apply(lambda row: label_facies(row, facies_labels), axis=1)
facies_label.describe()

sns.set()
sns.pairplot(facies_label.drop(['Cls1','Facies','Formation','Depth','NM_M','RELPOS'],axis=1),
             hue='FaciesLabels', palette=facies_color_map,
             hue_order=list(reversed(facies_labels)))

#switch back to default matplotlib plot style
mpl.rcParams.update(inline_rc)