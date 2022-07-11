# utilities used throughout the project

import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
import torch
from PIL import Image
import matplotlib.pyplot as plt

# CONSTANTS
VGG_MEAN = torch.tensor([0.485, 0.456, 0.406])
# choose not to normalize by scaling by making std a tensor of ones
VGG_STD = torch.tensor([1, 1, 1])


class StyleLoss(nn.Module):
    def __init__(self, target_gram):
        super(StyleLoss, self).__init__()
        self.target_gram = get_gram_matrix(target_gram)

    def forward(self, gen_feature):
        gram = get_gram_matrix(gen_feature)
        self.loss = nn.functional.mse_loss(gram, self.target_gram)
        return gen_feature


class ContentLoss(nn.Module):
    def __init__(self, target_feature=None):
        super(ContentLoss, self).__init__()
        self.target_feature = target_feature

    def forward(self, gen_feature):
        self.loss = nn.functional.mse_loss(gen_feature, self.target_feature)
        return gen_feature


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
            for layer in content_features:
                content_features[layer] = content_features[layer].detach()
            for layer in style_features:
                style_features[layer] = style_features[layer].detach()

        # what we actually need from the feature maps of the style image is its gram matrices
        gram_matrices = {}
        for layer in style_features:
            gram_matrices[layer] = get_gram_matrix(style_features[layer])

        return content_features, gram_matrices

    return get_features


def prepare_image(path, img_size=256):
    img = Image.open(path)

    # maybe remove normalization? because results are not good so far!
    convert_tensor = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=VGG_MEAN, std=VGG_STD),
        # somehow multipying by some large constant makes the results converge faster
        transforms.Lambda(lambda x: x.mul_(255.0))
    ])

    return convert_tensor(img).unsqueeze(0)


def display_image(image):
    img = image.clone()
    img = img.to("cpu")
    img = img.squeeze(0)
    # denormalizing
    img.div_(300.0)
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
    img_to_save.div_(255.0)
    img_to_save *= VGG_STD.view(3, 1, 1)
    img_to_save += VGG_MEAN.view(3, 1, 1)
    save_image(img_to_save, path)
