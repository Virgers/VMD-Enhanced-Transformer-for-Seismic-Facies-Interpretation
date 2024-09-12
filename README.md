# Time Series Library (TSlib)
We fork the THUML time-series library for solving the seismic facies interpretation task. Specifically, we utilize the open-source data F3 and Parihaka dataset to verify our improved Label-integrated and VMD-augmented transformer.

// ## Usage

1. Install Python 3.8. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare Data. You can obtain the well pre-processed datasets from [[Google Drive]](XXXX) 

<p align="center">
<img src=".\pic\dataset.png" height = "200" alt="" align=center />
</p>

3. Train and evaluate model. We provide the experiment scripts for seismic classification using various models:

```
# classification
bash ./scripts/classification/TimesNet.sh
```

4. Develop your own model.

- Add the model file to the folder `./models`. You can follow the `./models/Transformer.py`.
- Include the newly added model in the `Exp_Basic.model_dict` of  `./exp/exp_basic.py`.
- Create the corresponding scripts under the folder `./scripts`.

## Acknowledgement

We appreciate the THUML for this project : https://github.com/thuml/Time-Series-Library //
