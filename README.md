# VMD-Enhanced-Transformer-for-Facies-Interpretation

## Table of Contents
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Usage](#usage)
- [Attention](#attention)
- [Citation](#citation)
- [Contact](#contact)
- [Acknowledgement](#acknowledgement)

## Introduction
We adpat several base models from THUML [Time-Series-Library](https://pan.baidu.com/s/1wydQRBNdyylJZAvxCMjOPA) to adpat into our classification tasks with following approaches.

1. We propose **Label-Integrated Embedding** to add very few labels during embedding stage to guide the self-attention stage. Typically, this approach will enhance most time-series transformer model, and it depends on corresponding model embedding structure. <br><br>
One can check the Label-Integragted Embedding within ``./layers/Embed.py``. We provide two cases for this approach, one for having VMD the other one for without VMD.<br>

2. We introduce **VMD-Augmentation** approach to enlarge the tranining dataset. Basically, it decompose the original singals into multiple orthogonal components. The generated modes have more regular structures, which makes the model more easiy to converage.<br><br>
We present a simple snipet for generating in our case within `./utils/generate_VMD.py`. One can generate their own VMD data by selecting suitable parameters. We provide our vmd data in Baidu Drive.

3. In this project, we evaluate the model performances on two datasets. Therefore, I define two sets data provider for two datasets particularly. Within each data provider, it provides two situations having vmd-assited or not.

4. We also testify the pre-training model effectiveness on down-streaming tasks with different masks and masking data volume. You can find the ssl parts in this project.

## Key Features
1. Introduce time series transformer to facies classification.
2. Introduce VMD-augmentation to process seismic data.
3. Introduce label-integrated embedding to time series transformer.

## Usage

1. Clone this repository. 

2. Install Python 3.8. For convenience, execute the following command. (Not recommend). 

    ```pip install -r requirements.txt```


3. Prepare Data. 
- *F3 facies dataset*
You can obtain the F3 facies datasets from [[Google Drive]](https://drive.google.com/drive/folders/124tphRV1eEtpxTiRSj6eE41xg2kgd7PW?usp=drive_link) or [[Baidu Drive]](https://pan.baidu.com/s/1wydQRBNdyylJZAvxCMjOPA) code: `f3fd`, Then place the downloaded data in the folder `./root_path` with `./data_path` and `./label_path` so forth. We also include the corresponding VMD data.

- *Paraihaka facies dataset*
You can obtain the Newzealand Pariahaka datasets from [[Google Drive]](https://drive.google.com/drive/folders/1AgxeoEeFYI0lC3cLWtXw3VyvAHoa2xTh?usp=drive_link) or [[Baidu Drive]](https://pan.baidu.com/s/1QNjanQDfN3H9JvOpoX_aYw) code: `NZfd`, Then place the downloaded data in the folder `./root_path` with `./data_path` and `./label_path` so forth. 

4. To debug. We modify arguments in every the `XXX_options.py` within Options class.

5. To explore the ssl, one can refer to `/utils/model_selecting_strategy`, and related code files with `ssl` in their names.

## Attention

1. Currently, we only provide the complete code for iTransformer, i.e., one can check the label-integrated embedding` and 'VMD-augmented' in iTransformer case. We'll complete other model cases ASAP.

2. The paper can be early [accessed](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10707629).



## Citation

If you find this repo useful, please cite our paper.
[Seismic Facies Classification Using Label-Integrated and VMD-Augmented Transformer](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10707629)


## Contact
If you have any questions or suggestions, feel free to contact: [Jinlong Huo](jinlong.huo99@gmail.com) Or describe it in Issues.

## Acknowledgement

This library is constructed based on the following repos:

- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)

All the experiment datasets are public, and we obtain them from the following links:

- [F3 facies dataset](https://github.com/yalaudah/facies_classification_benchmark.)
- [New Zealand Parihaka dataset](https://www.aicrowd.com/challenges/seismic-facies-identification-challenge)
