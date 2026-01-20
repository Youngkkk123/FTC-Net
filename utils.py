try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch 
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import random


# 标签增强
def privilege_preprocess(inputs_privilege, device, augmentation=True, aug_rate=0.3):
    inputs_privilege = F.one_hot(inputs_privilege, num_classes=2).float()
    if augmentation:
        label_aug = torch.rand(inputs_privilege.size(0)).to(device) * aug_rate
        inputs_privilege[:,0] = inputs_privilege[:,0] - label_aug
        inputs_privilege[:,1] = inputs_privilege[:,1] - label_aug
        inputs_privilege = torch.abs_(inputs_privilege)
    return inputs_privilege


def get_loss_weight(labels):
    sample_size = labels.size(0)
    p_samples = torch.sum(labels)
    n_samples = sample_size - p_samples
    p_weight = sample_size / (p_samples + 1)
    n_weight = sample_size / (n_samples + 1)
    loss_weight = labels * p_weight + (1-labels) * n_weight
    return loss_weight


# 展示并保存增强后的图像
def show_and_save_augmented_images(images, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 创建一个新的图像网格
    grid = vutils.make_grid(images, nrow=4, padding=2, normalize=True)

    # 展示图像
    plt.figure(figsize=(12, 20))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.title('Augmented Images')
    plt.savefig(save_path)
    # plt.show()
    plt.close()


def get_batch_n_images(trainloader, n,model_save_path):
    dataiter = iter(trainloader)  # 创建DataLoader迭代器
    for _ in range(n):
        batch = next(dataiter)
    images = batch['image']
    # image = images[0]
    # print(image.shape)
    show_and_save_augmented_images(images, os.path.join(model_save_path, 'batch{}_images.png'.format(n)))


def freeze_and_unfreeze_model(model, unfreeze_layers=None):
    if unfreeze_layers is None:
        unfreeze_layers = []  # 默认不解冻

    # 冻结所有层
    for param in model.parameters():
        param.requires_grad = False

    # 解冻指定的 encoder layers
    for layer_idx in unfreeze_layers:
        if 0 <= layer_idx < len(model.encoder.siglip_model.vision_model.encoder.layers):
            for param in model.encoder.siglip_model.vision_model.encoder.layers[layer_idx].parameters():
                param.requires_grad = True

    # 解冻 post_layernorm 和 head
    for param in model.encoder.siglip_model.vision_model.post_layernorm.parameters():
        param.requires_grad = True

    for param in model.encoder.siglip_model.vision_model.head.parameters():
        param.requires_grad = True

    # 解冻最后的两个全连接层和 dropout 层
    for param in model.fc1.parameters():
        param.requires_grad = True

    for param in model.dropout.parameters():
        param.requires_grad = True

    for param in model.fc2.parameters():
        param.requires_grad = True

    return model

def freeze_and_unfreeze_LoRAmodel(model, unfreeze_layers=None):
    if unfreeze_layers is None:
        unfreeze_layers = []  # 默认不解冻

    # 冻结所有层
    for param in model.parameters():
        param.requires_grad = False

    # 解冻指定的 encoder layers
    for layer_idx in unfreeze_layers:
        if 0 <= layer_idx < len(model.encoder.siglip_model.vision_model.encoder.layers):
            for param in model.encoder.siglip_model.vision_model.encoder.layers[layer_idx].parameters():
                param.requires_grad = True

    # 解冻 post_layernorm 和 head
    for param in model.encoder.siglip_model.vision_model.post_layernorm.parameters():
        param.requires_grad = True

    for param in model.encoder.siglip_model.vision_model.head.parameters():
        param.requires_grad = True

    # 解冻最后的两个全连接层和 dropout 层
    for param in model.fc1.parameters():
        param.requires_grad = True

    for param in model.dropout.parameters():
        param.requires_grad = True

    for param in model.fc2.parameters():
        param.requires_grad = True

    return model

def extract_patient_identifier(patient_path):
    parts = patient_path.split('\\')
    if len(parts) >= 4:
        return '\\'.join(parts[-3:-1])  # 提取倒数第4到倒数第2部分
    return patient_path  # 如果格式不符合预期，返回原始路径


class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, datasets, batch_size):
        self.datasets = datasets
        self.batch_size = batch_size
        self.pos_indices = []
        self.neg_indices = []

        # 遍历所有数据集，获取正样本和负样本的索引
        start_index = 0
        for dataset in self.datasets:
            labels = dataset.labels
            for i, label in enumerate(labels):
                if label == 1:
                    self.pos_indices.append(start_index + i)
                else:
                    self.neg_indices.append(start_index + i)
            start_index += len(labels)

        self.num_batches = min(len(self.pos_indices), len(self.neg_indices)) // (self.batch_size // 2)

    def __iter__(self):
        for _ in range(self.num_batches):
            pos_batch = random.sample(self.pos_indices, self.batch_size // 2)
            neg_batch = random.sample(self.neg_indices, self.batch_size // 2)
            yield pos_batch + neg_batch

    def __len__(self):
        return self.num_batches

