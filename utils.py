# utilities used throughout the project

# Plan: move the content and style loss functions here (and get gram matrix function)
# Structure the code in artistic_neural_net.py to have a train function that returns
#  a generated picture

import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
import torch
from PIL import Image
import matplotlib.pyplot as plt

# CONSTANTS
VGG_MEAN = torch.tensor([0.485, 0.456, 0.406])
VGG_STD = torch.tensor([0.229, 0.224, 0.225])
# the following list is the style conv layers used in paper
STYLE_NODES_CONV = ["features.0", "features.5",
                    "features.10", "features.19", "features.28"]
# I found that relu layers generally converges significatly faster
STYLE_NODES_RELU = ["features.1", "features.6",
                    "features.11", "features.20", "features.29"]
CONTENT_NODES = ["features.21"]


def content_loss(featmaps_gen, featmaps_target):
    # using the mean error seems to consistently work better than just the sum
    return nn.functional.mse_loss(featmaps_gen, featmaps_target, reduction="mean")


def style_loss(gram_matrix_gen, gram_matrix_target):
    return nn.functional.mse_loss(gram_matrix_gen, gram_matrix_target, reduction="sum")


def get_gram_matrix(featmaps):
    _, N, M_x, M_y = featmaps.shape
    M = M_x * M_y
    featmaps = featmaps.view(N, M)
    return (featmaps @ featmaps.T).div(N * M)


def get_content_and_style(content_feature_extractor, style_feature_extractor):

    def get_features(content_img, style_img, detach=False):
        # gets the features of the content image and the gram matrices of the style
        content_features = content_feature_extractor(content_img)
        style_features = style_feature_extractor(style_img)
        # making sure that the target content and style features are detached
        if detach:
            for layer in content_features.keys():
                content_features[layer] = content_features[layer].detach()
            for layer in style_features.keys():
                style_features[layer] = style_features[layer].detach()

        # what we actually need from the feature maps of the style image is its gram matrices
        gram_matrices = {}
        for layer in style_features.keys():
            gram_matrices[layer] = get_gram_matrix(style_features[layer])

        return content_features, gram_matrices

    return get_features


def prepare_image(path, img_size=256):
    img = Image.open(path)

    # maybe remove normalization? because results are not good so far!
    convert_tensor = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=VGG_MEAN, std=VGG_STD)
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
