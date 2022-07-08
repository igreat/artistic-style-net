# TODO: Make the code more modularized (split functionality over files where possible)
# TODO: Implement it with VGG networks or allow
#  functionality to switch between networks more easily

import torch
from torchvision.models import vgg19
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def content_loss(featmaps_gen, featmaps_target):
    return nn.functional.mse_loss(featmaps_gen, featmaps_target, reduction="sum")


def style_loss(gram_matrix_gen, gram_matrix_target):
    return 1/5 * nn.functional.mse_loss(gram_matrix_gen, gram_matrix_target, reduction="sum")


def get_gram_matrix(featmaps):
    _, N, M_x, M_y = featmaps.shape
    M = M_x * M_y
    featmaps = featmaps.view(N, M)
    return (featmaps @ featmaps.T).div(N * M)


# # loading pretrained model
model = vgg19(pretrained=True).eval().to(device)
# print(model)
# print(get_graph_node_names(model))
model.requires_grad = False
style_nodes = ["features.0", "features.5", "features.10", "features.19", "features.28"]

# # setting up feature extractors
style_feature_extractor = create_feature_extractor(model, return_nodes=style_nodes).to(device)
content_feature_extractor = create_feature_extractor(model, return_nodes=["features.21"]).to(device)

# loading test images
night = Image.open("night.jpg")
house = Image.open("house.jpg")

image_size = 512 # make sure both images have the same dimensions

vgg_mean_255 = torch.tensor([123.675, 116.28, 103.53]).to(device)
vgg_std = torch.tensor([1, 1, 1]).to(device)

convert_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((image_size, image_size)),
    # experimenting with range 255 (seems to work better for now)
    transforms.Lambda(lambda x: x.mul(255)),
    transforms.Normalize(mean=vgg_mean_255, std=vgg_std)
])

night = convert_tensor(night).view(1, 3, image_size, image_size).to(device)
house = convert_tensor(house).view(1, 3, image_size, image_size).to(device)

# initializing the input image to the content image
gen_image = torch.rand(1, 3, image_size, image_size).to(device).div(255/2) + 255/4

content_features_target = content_feature_extractor(house)
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

# setting up for optimization
content_weight = 1
style_weight = 1e6
gen_image.requires_grad = True
optimizer = optim.LBFGS([gen_image], max_iter=2000, line_search_fn='strong_wolfe')
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
    for layer in gen_features_content.keys():
        l_content += content_loss(gen_features_content[layer], content_features_target[layer])

    with torch.no_grad():
        if step_cnt % 50 == 0:
            print(
                f"step {step_cnt} - content loss: {content_weight * l_content} || style loss: {style_weight * l_style}")

    loss = style_weight * l_style + content_weight * l_content
    loss.backward()

    step_cnt += 1
    return loss


optimizer.step(closure)


# code to display the generated image and compare it with the style and content images
def display_image(image):
    img = image.clone()
    # "unnormalizing"
    img *= vgg_std.view(1, 3, 1, 1)
    img += vgg_mean_255.view(1, 3, 1, 1)
    img = img.div(255)
    img = img.squeeze(0)
    img = img.permute((1, 2, 0))
    img = img.to("cpu")
    plt.imshow(img)


gen_image.requires_grad = False

plt.figure()
display_image(house)

plt.figure()
display_image(night)

plt.figure()
display_image(gen_image)