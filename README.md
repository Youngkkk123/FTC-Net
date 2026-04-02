# FTC-Net

This is the official implementation of FTC-Net in our work. 

File descriptions are as follows:

| File Name                  | Function Description                                  |
| -------------------------- | ----------------------------------------------------- |
| `train_patient_FTC_net.py` | Model training                                        |
| `eval_patient_FTC_net.py`  | Model evaluation                                      |
| `datagenerator.py`         | Dataset class definition                              |
| `data_augmentation.py`     | Data augmentation                                     |
| `Loss.py`                  | Focal Loss function definition                        |
| `utils_index.py`           | Tool class for AUC (Area Under the Curve) calculation |

How to start
=============

This code can be easily performed on B-mode Ultrasound data with classification annotation. Here, we split the whole process into 4 steps so that you can reproduce the FTC-Net on your personal side.

* Step 1: Environment setting.
* Step 2: Data preparation.
* Step 3: Model training. 
* Step 4: Model evaluation. 

Step 1: Environment setting
-------------

Our code was implemented in following environment：

* Python == 3.9.0
* torch == 2.0.0
* cuda == 11.8
* transformer == 4.45.2
* GPU == NVIDIA L40 48GB

Step 2: Data preparation
-------------

Prepare the data according to our paper, make sure that the data for each patient comprise the required number of B-mode Ultrasound image crops.
You need to split the patient data into four parts: two training subsets (a positive training subset and a negative training subset), one validation set, and one test set.
Then, save the data path and the corresponding label in a `.npy` file for each set. 

For example:

| Name            | Format example                                               |
| --------------- | ------------------------------------------------------------ |
| `train_pos.npy` | `[["train_pos_path_1", "1"], ["train_pos_path_2", "1"], ..., ["train_pos_path_N", "1"]]` |
| `train_neg.npy` | `[["train_neg_path_1", "0"], ["train_neg_path_2", "0"], ..., ["train_neg_path_N", "0"]]` |
| `valid.npy`     | `[["valid_path_1", "1"], ["valid_path_2", "0"], ..., ["valid_path_N", "0"]]` |
| `test.npy`      | `[["test_path_1", "1"], ["test_path_2", "0"], ..., ["test_path_N", "1"]]` |


Step 3: Model training
-------------

Run the script `train_patient_FTC_net.py` to train FTC-Net using `train_pos.npy` , `train_neg.npy`and `valid.npy` files. The `train_pos.npy`  and `train_neg.npy` contain training set for model training and the `valid.npy` contains validation set for model selection.

Modify the code kind of `data_path` to your actual data save path, and modify `model_save_path` to your expected model save path.

During model training, the checkpoints of the best performance on the validation set are saved in the folder `model_save_path`.

## Step 4: Model evaluation

Run the script `eval_patient_FTC_net.py` to test the performance of FTC-Net using `test.npy`. The `test.npy` contains test set for model testing.

Before running the script, you need to set the model path to the training saved model path `model_save_path` and modify the code kind of `data_path` to your actual data save path.

Finally, you can obtain the ROC and corresponding AUC of FTC-Net on the test set，which are saved in `save_path`.

## Prospective Validation

### Dataset Overview

We prospectively collected **10 cases of follicular thyroid neoplasms** confirmed by surgical pathology, including:

- 5 cases of Follicular Thyroid Adenoma (FTA)
- 5 cases of Follicular Thyroid Carcinoma (FTC)

### Prospective Validation Results

The model achieved **90% accuracy** in prospective validation:

- Correct predictions: 9 cases
- Misclassification: 1 FTA case was incorrectly identified as FTC

### Data and Model Access

#### 1. Prospective Dataset

- Raw prospective data files are stored in the `prospective_data/` directory
- Case list (patient paths, labels) is saved in `prospective_data.npy`

#### 2. Trained Model

The pre-trained classification model can be downloaded via the link below for reproducing the prospective validation results:

- Download link: https://mypikpak.com/s/VOpBZa6m-FOvmDekh8Dgo9oIo2
- Extraction key: `2y9f`

#### 3. Evaluation Instructions

To evaluate the model on the prospective dataset:

1. Download the model weights `model.pt` from the above link

2. Place the weights file in the project root directory (or update the model path in `prediction.py`)

3. Run the inference script: `python prediction.py`
