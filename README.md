# VMD-Enhanced-Transformer-for-Facies-Interpretation
We adpat several base models from TSlib to adpat into our classification tasks.

1. We propose **label-integrated embedding** to add very few labels during embedding stage to guide the self-attention stage. Typically, this approach will enhance most time-series transformer model, and it depends on corresponding model embedding structure.
2. We introduce **VMD-augmented** approach to enlarge the tranining dataset.

## Module examination

We evalutate the module effectiveness by implementing several cutting-edge time-series transformers. We list the original paper and codes that we use in our paper.


## Usage

1. Install Python 3.8. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare Data. 
- *F3 facies dataset*
You can obtain the F3 facies datasets from [[Google Drive]]() or [[Baidu Drive]](https://pan.baidu.com/s/1wydQRBNdyylJZAvxCMjOPA) code: `f3fd`, Then place the downloaded data in the folder `./root_path` with `./data_path` and `./label_path` so forth. Here is a summary of supported datasets.

- *Paraihaka facies dataset*
You can obtain the Newzealand Pariahaka datasets from [[Google Drive]]() or [[Baidu Drive]](https://pan.baidu.com/s/1QNjanQDfN3H9JvOpoX_aYw) code: `NZfd`, Then place the downloaded data in the folder `./root_path` with `./data_path` and `./label_path` so forth. Here is a summary of supported datasets.

I define two sets data provider for two datasets particularly. Within each data provider, it provides two situations having vmd-assited or not.


3. Train and evaluate model. We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```
# long-term forecast
bash ./scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh
# short-term forecast
bash ./scripts/short_term_forecast/TimesNet_M4.sh
# imputation
bash ./scripts/imputation/ETT_script/TimesNet_ETTh1.sh
# anomaly detection
bash ./scripts/anomaly_detection/PSM/TimesNet.sh
# classification
bash ./scripts/classification/TimesNet.sh
```
4. To debug the model. We modify arguments in every the `XXX_options.py` within Options class.

5. Check label-integrated embedding.

- We .

6. Check the VMD-augmented embedding.

## Citation

If you find this repo useful, please cite our paper.

```
@inproceedings{wu2023timesnet,
  title={TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  author={Haixu Wu and Tengge Hu and Yong Liu and Hang Zhou and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Learning Representations},
  year={2023},
}
```

## Contact
If you have any questions or suggestions, feel free to contact:

- Jinlong Huo (jinlong.huo99@gmail.com)

Or describe it in Issues.

## Acknowledgement

This library is constructed based on the following repos:

- Time-Series-Library: https://github.com/thuml/Time-Series-Library.

- Forecasting: https://github.com/thuml/Autoformer.

- Anomaly Detection: https://github.com/thuml/Anomaly-Transformer.

- Classification: https://github.com/thuml/Flowformer.

All the experiment datasets are public, and we obtain them from the following links:

- F3 facies dataset: https://github.com/yalaudah/facies_classification_benchmark.
- New Zealand Parihaka dataset: https://www.aicrowd.com/challenges/seismic-facies-identification-challenge. 
