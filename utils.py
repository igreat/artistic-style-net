# utilities used throughout the project

import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
import torch
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import numpy as np
from PIL import Image


# Consider making normalization part of the feature extractor in models
# CONSTANTS
VGG_MEAN = torch.tensor([0.485, 0.456, 0.406])
# this is not really the vgg std but a way to scale the inputs as part of the normalization
VGG_STD = torch.tensor([1/255, 1/255, 1/255])


class StyleLoss(nn.Module):
    def __init__(self, target_gram, weight):
        super(StyleLoss, self).__init__()
        self.target_gram = get_gram_matrix(target_gram).detach()
        self.weight = weight * 5e-2

    def forward(self, gen_feature):
        gram_matrix = get_gram_matrix(gen_feature)
        self.loss = self.weight * \
            F.mse_loss(gram_matrix, self.target_gram)
        return gen_feature


class ContentLoss(nn.Module):
    def __init__(self, target_feature, weight):
        super(ContentLoss, self).__init__()
        self.weight = weight
        self.target_feature = target_feature.detach()

    def forward(self, gen_feature):
        self.loss = self.weight * \
            F.mse_loss(gen_feature, self.target_feature, reduction="sum")
        return gen_feature


def get_gram_matrix(featmaps):
    _, c, h, w = featmaps.shape
    featmaps = featmaps.view(c, h * w)
    return (featmaps @ featmaps.T).div(h * w)


# Total variation loss
class TVLoss(nn.Module):
    def __init__(self, weight):
        super(TVLoss, self).__init__()
        self.weight = weight

    # consider changing the abs to square
    def forward(self, featmaps):
        self.x_diff = featmaps[:, :, 1:, :] - featmaps[:, :, :-1, :]
        self.y_diff = featmaps[:, :, :, 1:] - featmaps[:, :, :, :-1]
        self.loss = self.weight * (torch.sum(torch.abs(self.x_diff)) +
                                   torch.sum(torch.abs(self.y_diff)))
        return featmaps


def match_color(input_img, color_img):

    # returns the input_img but with the same color distribution as color_img
    _, input_h, input_w = input_img.shape
    _, color_h, color_w = color_img.shape

    input_img = input_img.to("cpu").view(3, input_h * input_w).numpy()
    color_img = color_img.to("cpu").view(3, color_h * color_w).numpy()

    cov_input = np.cov(input_img)
    cov_color = np.cov(color_img)

    A = sqrtm(cov_color) @ np.linalg.inv(sqrtm(cov_input))

    # applying the transformation
    input_img = A @ (input_img - np.mean(input_img, axis=1).reshape(-1, 1)) + \
        np.mean(color_img, axis=1).reshape(-1, 1)

    return torch.tensor(input_img.reshape(3, input_h, input_w), dtype=torch.float)


def process_image(path, img_size=256, is_color_matched=False, color_img=None):

    image = Image.open(path)

    convert_tensor = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=VGG_MEAN, std=VGG_STD)
    ])

    tensor = convert_tensor(image)
    if is_color_matched:
        tensor = match_color(tensor, color_img.squeeze(0))

    return tensor.unsqueeze(0)


def deprocess_image(tensor):
    tensor = tensor.clone().to("cpu")
    tensor *= VGG_STD.view(3, 1, 1)
    tensor += VGG_MEAN.view(3, 1, 1)
    return tensor


def display_image(image):
    img = deprocess_image(image)
    img = img.permute((1, 2, 0))
    plt.imshow(img)


def save_img(gen_img, path="generated images/untitled.png"):
    # saving image
    img_to_save = deprocess_image(gen_img)
    save_image(img_to_save, path)
