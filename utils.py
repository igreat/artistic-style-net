# utilities used throughout the project

import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
import torch
from PIL import Image
import matplotlib.pyplot as plt


# Consider making normalization part of the feature extractor in models
# CONSTANTS
VGG_MEAN = torch.tensor([0.485, 0.456, 0.406])
# this is not really the vgg std but a way to scale the inputs as part of the normalization
VGG_STD = torch.tensor([1/255, 1/255, 1/255])


class StyleLoss(nn.Module):
    def __init__(self, target_gram, weight):
        super(StyleLoss, self).__init__()
        self.target_gram = get_gram_matrix(target_gram).detach()
        self.weight = weight

    def forward(self, gen_feature):
        gram_matrix = get_gram_matrix(gen_feature)
        self.loss = self.weight * F.mse_loss(gram_matrix, self.target_gram)
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


def prepare_image(path, img_size=256):
    img = Image.open(path)

    # maybe remove normalization? because results are not good so far!
    convert_tensor = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=VGG_MEAN, std=VGG_STD),
    ])

    return convert_tensor(img).unsqueeze(0)


def display_image(image):
    img = image.clone()
    img = img.to("cpu")
    img = img.squeeze(0)
    # denormalizing
    img *= VGG_STD.view(3, 1, 1)
    img += VGG_MEAN.view(3, 1, 1)
    img = img.permute((1, 2, 0))
    img = img.to("cpu")
    plt.imshow(img)


def save_img(gen_img, path="generated images/untitled.png"):
    # saving image
    img_to_save = gen_img.clone()
    img_to_save = gen_img.to("cpu").squeeze(0)
    # denormalizing
    img_to_save *= VGG_STD.view(3, 1, 1)
    img_to_save += VGG_MEAN.view(3, 1, 1)
    save_image(img_to_save, path)
