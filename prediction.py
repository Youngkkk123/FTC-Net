import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from utils import extract_patient_identifier
from transformers import AutoModel, AutoProcessor
from peft import LoraConfig, get_peft_model
from torch.backends import cudnn
from torch.utils.data import Dataset
from data_augmentation import get_data_transform_2D_vit
import pandas as pd
import random

# Set random seed for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
os.environ['PYTHONHASHSEED'] = str(0)
cudnn.benchmark = False
cudnn.deterministic = True


class Datasets_Bmode_patient_dir(Dataset):
    def __init__(self, data_list, transform=None, device='cuda:0'):
        self.data_list = data_list
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # 提取图像路径和标签
        patient_path, label = self.data_list[index]
        patient_path = os.path.join('./prospective_data', patient_path, 'ROI_image')
        label = int(label)

        data = {
            'path': patient_path,
            'label': label
        }
        return data


def Square_Generated(image,fill_style:int=0,map_color:tuple=0):
    w, h = image.size
    new_image = Image.new(image.mode, size=(max(w, h), max(w, h)),color=map_color)
    if fill_style == 0:
        point = int(abs(w - h)) // 2
        box = (point,0) if w < h else (0,point)
    elif fill_style == 1:
        length = int(abs(w - h))  # 一侧需要填充的长度
        box = (length, 0) if w < h else (0, length)  # 放在box中
    else:
        box = (0,0)
    new_image.paste(image, box)
    return new_image


def validate_patient(model, dataloader, transform, processor, device):
    """
    Validate model performance on patient-level data
    Args:
        model: Trained model for thyroid tumor classification
        dataloader: DataLoader for patient-level dataset
        transform: Image transformation pipeline for validation
        processor: Processor for SigLIP model (image/text preprocessing)
        device: Computing device (cuda/cpu)

    Returns:
        patient_labels: True labels of all patients
        patient_image_probs: Prediction probabilities of all images per patient
        patient_image_names: Filenames of all images per patient
        patient_identifiers: Unique identifiers of all patients
    """
    with torch.no_grad():
        model.eval()

        patient_labels = []
        patient_image_probs = []
        patient_image_names = []
        patient_identifiers = []

        # Define class texts for text-image contrastive learning
        class_texts = [
            "An image of thyroid adenoma.",
            "An image of thyroid follicular carcinoma."
        ]

        # Preprocess text inputs and get text embeddings
        text_inputs = processor(
            text=class_texts,
            return_tensors="pt",
            padding="max_length"
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        text_embeds = model.get_text_features(**text_inputs)
        text_embeds = F.normalize(text_embeds, p=2, dim=1)  # L2 normalization

        # Iterate over each patient in dataloader
        for _, batch_data in enumerate(dataloader):
            patient_folder = batch_data['path'][0]
            label = batch_data['label'].item()

            # Extract unique identifier for current patient
            patient_id = extract_patient_identifier(patient_folder)
            patient_identifiers.append(patient_id)

            # Get all image paths in patient folder (support common image formats)
            image_paths = [
                os.path.join(patient_folder, fname)
                for fname in os.listdir(patient_folder)
                if os.path.isfile(os.path.join(patient_folder, fname)) and
                   fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]

            # Handle case with no images in patient folder
            if not image_paths:
                patient_image_probs.append([])
                patient_image_names.append([])
                patient_labels.append(label)
                continue

            # Store probabilities and filenames for current patient's images
            cur_patient_probs = []
            cur_patient_names = []

            # Process each image of the current patient
            for img_path in image_paths:
                img_name = os.path.basename(img_path)

                # Load and preprocess image
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img = Square_Generated(img)  # Generate square image
                if transform:
                    img = transform(img)

                # Preprocess image and get image embeddings
                image_inputs = processor(images=img, return_tensors="pt").to(device)
                image_embeds = model.get_image_features(**image_inputs)
                image_embeds = F.normalize(image_embeds, p=2, dim=1)  # L2 normalization

                # Calculate cosine similarity between text and image embeddings
                cos_sim = (
                        torch.matmul(text_embeds.detach().clone(), image_embeds.t()) *
                        model.base_model.logit_scale.exp() + model.base_model.logit_bias
                ).T

                # Calculate prediction probability for positive class
                prob = torch.softmax(cos_sim, dim=1)[:, 1].item()
                cur_patient_probs.append(prob)
                cur_patient_names.append(img_name)

            # Update patient-level lists
            patient_labels.append(label)
            patient_image_probs.append(cur_patient_probs)
            patient_image_names.append(cur_patient_names)

    return patient_labels, patient_image_probs, patient_image_names, patient_identifiers


if __name__ == '__main__':
    # Set computing device (CUDA if available, otherwise CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    patch_size = (224, 224)  # Input image size for model
    batch_size = 1  # Batch size (patient-level processing, 1 patient per batch)
    threshold = 0.473

    # Initialize model and processor (SigLIP base model)
    model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224", do_rescale=False)

    # model_path = r""
    # model = AutoModel.from_pretrained(model_path, local_files_only=True)
    # processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, do_rescale=False)

    # Configure LoRA (Low-Rank Adaptation) for vision and text encoders
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=['q_proj', 'k_proj', 'v_proj'],
        bias="none"
    )
    model.vision_model = get_peft_model(model.vision_model, lora_config)
    model.text_model = get_peft_model(model.text_model, lora_config)

    # Load trained model weights
    model_checkpoint = torch.load(
        f="./model.pt",
        map_location=device
    ).state_dict()
    model.load_state_dict(model_checkpoint, strict=True)
    model = model.to(device)

    img_transforms = get_data_transform_2D_vit(patch_size)

    # Define test files
    test_files = ['prospective_data.npy']  # Example: ['test_data.npy']

    # Process each test file
    for test_file in test_files:
        # Load test dataset (numpy format)
        test_data = np.load(test_file, allow_pickle=True)

        # Create dataset and dataloader for test data
        test_dataset = Datasets_Bmode_patient_dir(test_data, device=device)
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        # Run patient-level validation
        labels, all_probs, all_names, patient_ids = validate_patient(model, test_dataloader, img_transforms['val'], processor, device)

        # Prepare result data for Excel export
        patient_max_probs = [max(probs) if probs else None for probs in all_probs]
        # 根据固定阈值生成预测结果
        patient_preds = [1 if prob >= threshold else 0 if prob is not None else None for prob in patient_max_probs]

        result_data = {
            'identifier': patient_ids,
            'label': labels,
            'max_prob': patient_max_probs,
            'prediction': patient_preds
        }

        # Sort images by prediction probability (descending) for each patient
        sorted_probs = []
        sorted_names = []
        for probs, names in zip(all_probs, all_names):
            if probs and names:
                combined = sorted(zip(probs, names), key=lambda x: x[0], reverse=True)
                p, n = zip(*combined)
                sorted_probs.append(list(p))
                sorted_names.append(list(n))
            else:
                sorted_probs.append([])
                sorted_names.append([])

        # Add sorted image probabilities and filenames to result data
        max_img_count = max([len(p) for p in sorted_probs]) if sorted_probs else 0
        for i in range(max_img_count):
            result_data[f'image_{i + 1}_prob'] = [
                probs[i] if i < len(probs) else None for probs in sorted_probs
            ]
            result_data[f'image_{i + 1}_name'] = [
                names[i] if i < len(names) else None for names in sorted_names
            ]

        # Save results to Excel file
        df = pd.DataFrame(result_data)
        excel_filename = "model_results.xlsx"
        df.to_excel(excel_filename, sheet_name='results', index=False)