import os
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Loss import FocalLoss
from data_augmentation import get_data_transform_2D_vit
from utils_index import calcAUC, calcACCSENSPE
from sklearn.metrics import *
from datagenerator import Datasets_Bmode_patient_dir
import random
from PIL import Image
from transformers import AutoModel, AutoProcessor
from peft import LoraConfig, get_peft_model

# Set random seed for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
os.environ['PYTHONHASHSEED'] = str(0)
from torch.backends import cudnn

cudnn.benchmark = False
cudnn.deterministic = True


# Function to pad image to a square shape
def Square_Generated(image, fill_style: int = 0, map_color: tuple = 0):
    w, h = image.size  # Get image dimensions
    new_image = Image.new(image.mode, size=(max(w, h), max(w, h)), color=map_color)
    if fill_style == 0:
        point = int(abs(w - h)) // 2
        box = (point, 0) if w < h else (0, point)
    elif fill_style == 1:
        length = int(abs(w - h))  # Length to pad on one side
        box = (length, 0) if w < h else (0, length)  # Set paste coordinates
    else:
        box = (0, 0)
    new_image.paste(image, box)  # Paste original image onto square image, box is the top-left coordinate
    return new_image


def extract_patient_identifier(patient_path):
    parts = patient_path.split('\\')
    if len(parts) >= 4:
        return '\\'.join(parts[-4:-1])
    return patient_path


# Patient-level validation process
def val_patient_process(model, datasetloader, transform, processor, device):
    with torch.no_grad():
        model.eval()

        total_loss = 0
        patient_truth_list = []  # Store ground truth label for each patient
        patient_probability_list = []  # Store predicted probability for each patient

        # Define text prompts for the two classes
        class_texts = [
            "An image of thyroid adenoma.",
            "An image of thyroid follicular carcinoma."
        ]

        # Preprocess text and obtain text embeddings
        text_inputs = processor(
            text=class_texts,
            return_tensors="pt",
            padding="max_length"
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        # Extract and normalize text features
        text_outputs = model.get_text_features(**text_inputs)
        text_embeds = F.normalize(text_outputs, p=2, dim=1)  # Ensure L2 normalization

        # Initialize loss function
        criterion = FocalLoss(gamma=2, alpha=0.75)

        # Iterate over each patient
        for j, datas in enumerate(datasetloader):
            patient_path = datas['path'][0]
            label = datas['label'].item()

            # Get all image paths in the patient directory
            image_paths = [os.path.join(patient_path, f) for f in os.listdir(patient_path)
                           if os.path.isfile(os.path.join(patient_path, f)) and
                           f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

            if not image_paths:
                print(f"No images found in folder {patient_path}")
                continue

            # Store prediction probabilities for all images of the current patient
            patient_image_probs = []

            for img_path in image_paths:
                # Load and preprocess image
                img = Image.open(img_path).convert('L')
                img = Square_Generated(img)
                if transform:
                    img = transform(img)

                # Extract and normalize image features
                image_inputs = processor(images=img, return_tensors="pt").to(device)
                image_embeds = model.get_image_features(**image_inputs)
                image_embeds = F.normalize(image_embeds, p=2, dim=1)

                # Calculate cosine similarity between image and text embeddings
                cos_similarity = (
                        torch.matmul(text_embeds.detach().clone(),
                                     image_embeds.t().to(text_embeds.device)) * model.base_model.logit_scale.exp()
                        + model.base_model.logit_bias
                ).T

                # Calculate prediction probability
                probability = torch.softmax(cos_similarity, dim=1)[:, 1].item()
                patient_image_probs.append(probability)

                # Calculate loss
                label_tensor = torch.tensor([label], dtype=torch.long).to(device)
                loss = criterion(cos_similarity, label_tensor)
                total_loss += loss.item()

            # Use maximum probability as patient-level prediction
            patient_probability = max(patient_image_probs)

            patient_truth_list.append(label)
            patient_probability_list.append(patient_probability)

        # Calculate evaluation metrics
        avg_auc, threshold = calcAUC(patient_truth_list, patient_probability_list)
        acc, sen, spe = calcACCSENSPE(patient_truth_list, patient_probability_list, threshold)
        avg_loss = total_loss / len(patient_truth_list) if patient_truth_list else 0

    return avg_loss, avg_auc, acc, sen, spe, threshold, patient_truth_list, patient_probability_list


if __name__ == '__main__':
    # Set computing device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Configuration parameters
    patchsize = (224, 224)

    data_path = './data_npy'
    model_load_fold = './model_save'
    if os.path.exists(model_load_fold) is False:
        raise ValueError('{} is not exist.'.format(model_load_fold))

    save_path = './test_save'
    test_save_path = os.path.join(save_path)
    os.makedirs(test_save_path, exist_ok=True)

    # Load trained model checkpoint
    model_load_path = os.path.join(model_load_fold, 'Best_model.pt')

    # Initialize SigLIP model and processor
    model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device)
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224", do_rescale=False)

    # Load pre-trained SigLIP model and processor locally
    # model_path = r"D:\Work\Thyroid_classify\.local_models\siglip-base-patch16-224"
    # model = AutoModel.from_pretrained(model_path, local_files_only=True)
    # processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, do_rescale=False)

    # LoRA configuration
    config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=['q_proj', 'k_proj', 'v_proj'],
        bias="none"
    )

    # Apply LoRA to vision and text encoders
    model.vision_model = get_peft_model(model.vision_model, config)
    model.text_model = get_peft_model(model.text_model, config)

    # Load trained weights
    model_checkpoint = torch.load(model_load_path).state_dict()
    model.load_state_dict(model_checkpoint, strict=True)
    model = model.to(device)

    # Get data transformation pipeline
    img_trans = get_data_transform_2D_vit(patchsize)

    # Load test dataset
    test_path = os.path.join(data_path, "test.npy")
    test_list = np.load(test_path, allow_pickle=True)

    datasetTest = Datasets_Bmode_patient_dir(test_list, device=device)
    testloader = DataLoader(dataset=datasetTest, batch_size=1, shuffle=False)

    epoch_start = time.time()
    # Test set evaluation
    print("----------------------> test result <----------------------------")
    avg_test_loss, avg_test_auc, test_acc, test_sen, test_spe, test_threshold, test_truth_list, test_probability_list = val_patient_process(
        model, testloader, img_trans['val'], processor, device)

    # Save prediction results to Excel
    test_model_preds = {
        'identifier': [extract_patient_identifier(item) for item in test_list[:, 0]],
        'label': test_truth_list,
        'pro': test_probability_list
    }
    data = pd.DataFrame(test_model_preds)
    data.to_excel(os.path.join('{}/test_pre.xlsx'.format(test_save_path)), sheet_name='sheet1')

    # Save test metrics to text file
    test_info = "Test: Loss: {:.4f}, AUC: {:.4f}, ACC: {:.4f}, SEN: {:.4f}, SPE: {:.4f}, Threshold: {:.4f}".format(
        avg_test_loss, avg_test_auc, test_acc, test_sen, test_spe, test_threshold
    )
    with open(os.path.join('{}/test_result.txt'.format(test_save_path)), 'w') as f:
        f.write(test_info)
    print(test_info)

    # Plot and save ROC curve
    plt.figure(figsize=(8, 6), dpi=300)
    fpr, tpr, thresholds = roc_curve(test_truth_list, test_probability_list, pos_label=1)
    plt.plot(fpr, tpr, lw=2, label='(AUC = {:.3f})'.format(avg_test_auc))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('1-Specificity', fontsize=13)
    plt.ylabel('Sensitivity', fontsize=13)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, left=0.2)
    plt.legend(fontsize=12, loc='lower right')
    plt.savefig(os.path.join(test_save_path, 'test_auc.png'))
    plt.close()

    # Print total inference time
    epoch_end = time.time()
    print("all time: {:.4f} s".format(epoch_end - epoch_start))
