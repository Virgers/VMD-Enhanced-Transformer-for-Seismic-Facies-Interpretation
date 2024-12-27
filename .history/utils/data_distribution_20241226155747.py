inline_rc = dict(mpl.rcParams)

import seaborn as sns
import numpy as np

facies_label = np.load('/home/dell/disk1/Jinlong/faciesdata/train_label.npy')

sns.set()
sns.pairplot(facies_label.drop(['Cls1','Facies','Formation','Depth','NM_M','RELPOS'],axis=1),
             hue='FaciesLabels', palette=facies_color_map,
             hue_order=list(reversed(facies_labels)))

#switch back to default matplotlib plot style
mpl.rcParams.update(inline_rc)