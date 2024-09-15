# VMD-Enhanced-Transformer-for-Facies-Interpretation

## Introduction
We adpat several base models from THUML [Time-Series-Library](https://pan.baidu.com/s/1wydQRBNdyylJZAvxCMjOPA) to adpat into our classification tasks with following approaches.

1. We propose **Label-Integrated Embedding** to add very few labels during embedding stage to guide the self-attention stage. Typically, this approach will enhance most time-series transformer model, and it depends on corresponding model embedding structure. <br><br>
One can check the Label-Integragted Embedding within ``./layers/Embed.py``. We provide two cases for this approach, one for having VMD the other one for without VMD.<br>

2. We introduce **VMD-Augmentation** approach to enlarge the tranining dataset. Basically, it decompose the original singals into multiple orthogonal components. The generated modes have more regular structures, which makes the model more easiy to converage.<br><br>
We present a simple snipet for generating in our case within `./utils/generate_VMD.py`. One can generate their own VMD data by selecting suitable parameters. We provide our vmd data in Baidu Drive.

3. In this project, we evaluate the model performances on two datasets. Therefore, I define two sets data provider for two datasets particularly. Within each data provider, it provides two situations having vmd-assited or not.


## Attention
```
1. Currently, we only provide the complete code for iTransformer, i.e., one can check the label-integrated embedding` and 'VMD-augmented' in iTransformer case. We'll complete other model cases ASAP.
2. The paper is coming soon.
```


## Usage

1. Clone this repository. 

2. Install Python 3.8. For convenience, execute the following command. (Not recommend). 

```
pip install -r requirements.txt
```

2. Prepare Data. 
- *F3 facies dataset*
You can obtain the F3 facies datasets from [[Google Drive]]() or [[Baidu Drive]](https://pan.baidu.com/s/1wydQRBNdyylJZAvxCMjOPA) code: `f3fd`, Then place the downloaded data in the folder `./root_path` with `./data_path` and `./label_path` so forth. Here is a summary of supported datasets.

- *Paraihaka facies dataset*
You can obtain the Newzealand Pariahaka datasets from [[Google Drive]]() or [[Baidu Drive]](https://pan.baidu.com/s/1QNjanQDfN3H9JvOpoX_aYw) code: `NZfd`, Then place the downloaded data in the folder `./root_path` with `./data_path` and `./label_path` so forth. Here is a summary of supported datasets.

3. To debug. We modify arguments in every the `XXX_options.py` within Options class.


## Citation

If you find this repo useful, please cite our paper.

```
XXX
```

## Contact
If you have any questions or suggestions, feel free to contact:

- Jinlong Huo (jinlong.huo99@gmail.com)

Or describe it in Issues.

## Acknowledgement

This library is constructed based on the following repos:

- Time-Series-Library: https://github.com/thuml/Time-Series-Library.

All the experiment datasets are public, and we obtain them from the following links:

- F3 facies dataset: https://github.com/yalaudah/facies_classification_benchmark.
- New Zealand Parihaka dataset: https://www.aicrowd.com/challenges/seismic-facies-identification-challenge. 
