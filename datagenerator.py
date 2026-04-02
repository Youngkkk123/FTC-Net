from torch.utils.data import Dataset
import os
from PIL import Image


def Square_Generated(image,fill_style:int=0,map_color:tuple=0):
    w, h = image.size
    new_image = Image.new(image.mode, size=(max(w, h), max(w, h)),color=map_color)
    if fill_style == 0:
        point = int(abs(w - h)) // 2
        box = (point,0) if w < h else (0,point)
    elif fill_style == 1:
        length = int(abs(w - h))
        box = (length, 0) if w < h else (0, length)
    else:
        box = (0,0)
    new_image.paste(image, box)
    return new_image


class Datasets_Bmode_patient_dir(Dataset):
    def __init__(self, data_list, transform=None, device='cuda:0'):
        self.data_list = data_list
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # Extract image paths and labels
        patient_path, label = self.data_list[index]
        # patient_path = patient_path.replace('\\', '/')

        label = int(label)

        data = {
            'path': patient_path,
            'label': label
        }
        return data


class Datasets_Bmode_roi_patient_allimg(Dataset):
    def __init__(self, data_list, transform=None, device='cuda:0'):
        self.data_list = data_list
        self.transform = transform
        self.device = device
        self.image_paths = []
        self.labels = []

        for patient_folder_path, label in self.data_list:
            # patient_folder_path = patient_folder_path.replace('\\', '/')

            label = int(label)
            image_paths = [os.path.join(patient_folder_path, f)
                           for f in os.listdir(patient_folder_path)
                           if os.path.isfile(os.path.join(patient_folder_path, f))
                           and f.lower().endswith(('.png', '.jpg', '.jpeg', 'bmp'))]
            if not image_paths:
                raise ValueError(f"No images found in folder {patient_folder_path}")
            self.image_paths.extend(image_paths)
            self.labels.extend([label] * len(image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label = self.labels[index]

        img = Image.open(img_path).convert('L')
        img = Square_Generated(img)
        if self.transform:
            img = self.transform(img)

        return {
            'image': img,
            'label': label,
            'path': img_path
        }

