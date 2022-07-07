# TODO: Make the code more modularized (split functionality over files where possible)
# TODO: Implement it with VGG networks or allow
#  functionality to switch between networks more easily

import torch
from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def content_loss(featmaps_gen, featmaps_target):
    return nn.functional.mse_loss(featmaps_gen, featmaps_target)


def style_loss(gram_matrix_gen, gram_matrix_target):
    return nn.functional.mse_loss(gram_matrix_gen, gram_matrix_target)


def get_gram_matrix(featmaps):
    _, N, M_x, M_y = featmaps.shape
    M = M_x * M_y
    featmaps = featmaps.view(N, M)
    return featmaps @ featmaps.T


# # loading pretrained model
model = resnet50(pretrained=True).eval().to(device)
model.requires_grad = False
style_nodes = ["layer1", "layer2", "layer3", "layer4"]

# setting up feature extractors
style_feature_extractor = create_feature_extractor(model, return_nodes=style_nodes).to(device)
content_feature_extractor = create_feature_extractor(model, return_nodes=["layer3"]).to(device)


# loading test images
night = Image.open("images/night.jpg")
lion = Image.open("images/lion.jpg")

image_size = 512 # make sure both images have the same dimensions
convert_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((image_size, image_size))
])

night = convert_tensor(night).view(1, 3, image_size, image_size).to(device)
lion = convert_tensor(lion).view(1, 3, image_size, image_size).to(device)

# initializing the input image to the content image
gen_image = lion.clone()
gen_image = gen_image.to(device)

content_features_target = content_feature_extractor(lion)
style_features_target = style_feature_extractor(night)

# making sure that the target content and style features are detached from the computational graph
for layer in content_features_target.keys():
    content_features_target[layer] = content_features_target[layer].detach()

for layer in style_features_target.keys():
    style_features_target[layer] = style_features_target[layer].detach()

# what we actually need from the feature maps of the style image is its gram matrices
gram_matrices_target = {}
for layer in style_features_target.keys():
    gram_matrices_target[layer] = get_gram_matrix(style_features_target[layer])
# possibly refactor above code using dictionary comprehension

content_weight = 1
style_weight = 1e6
gen_image.requires_grad = True
optimizer = optim.LBFGS([gen_image], max_iter=500, line_search_fn="strong_wolfe")
step_cnt = 0


# generating the target image
def closure():
    global step_cnt
    optimizer.zero_grad()
    gen_features_content = content_feature_extractor(gen_image)
    gen_features_style = style_feature_extractor(gen_image)
    gram_matrices_gen = {}
    for layer in style_features_target.keys():
        gram_matrices_gen[layer] = get_gram_matrix(gen_features_style[layer])
    # possibly refactor above code using dictionary comprehension
    # or putting it in a separate function

    # getting the total loss
    l_style = 0
    l_content = 0
    # for now, I will let all considered layers contribute equally
    for layer in gram_matrices_target.keys():
        l_style += style_loss(gram_matrices_gen[layer], gram_matrices_target[layer])

    with torch.no_grad():
        for layer in gen_features_content.keys():
            l_content += content_loss(gen_features_content[layer], content_features_target[layer])
        print(f"step {step_cnt} - content loss: {content_weight * l_content} || style loss: {style_weight * l_style}")

    loss = style_weight * l_style + content_weight * l_content
    loss.backward()

    step_cnt += 1
    return loss


optimizer.step(closure)


# code to display the generated image and compare it with the style and content images
def display_image(image):
    image = image.to("cpu")
    image = image.squeeze(0)
    image = image.permute((1, 2, 0))
    plt.imshow(image)


gen_image.requires_grad = False

plt.figure()
display_image(lion)

plt.figure()
display_image(night)

plt.figure()
display_image(gen_image)
