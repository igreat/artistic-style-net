import torch
from torchvision.models import resnet50
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
    # featmaps_gen: the feature maps of the generated image
    #        shape: N * M
    # featmaps_target: the feature maps of the target content image
    #        shape: N * M
    # ** think about whether the mean in mse_loss is a good idea ** #
    # This is essentially summing up the squares of diff
    # and then dividing the sum by N * M (M: M_x * M_y)
    return nn.functional.mse_loss(featmaps_gen, featmaps_target)


def style_loss(gram_matrix_gen, gram_matrix_target):
    # This is essentially summing up the squares of diff
    # and then dividing the sum by N * N
    # perhaps the extra factor doesn't matter and constants
    # content and style weights are good enough for control?
    return nn.functional.mse_loss(gram_matrix_gen, gram_matrix_target)


def get_gram_matrix(featmaps):
    _, N, M_x, M_y = featmaps.shape
    M = M_x * M_y
    featmaps = featmaps.view(N, M)
    return featmaps @ featmaps.T


# # loading pretrained model
model = resnet50(pretrained=True).eval().to(device)
style_nodes = ["layer1", "layer2", "layer3", "layer4"]

# setting up feature extractors
style_feature_extractor = create_feature_extractor(model, return_nodes=style_nodes).to(device)
content_feature_extractor = create_feature_extractor(model, return_nodes=["layer3", "layer4"]).to(device)


# loading test images
fire = Image.open("images/fire.jpg")
lion = Image.open("images/lion.jpg")

image_size = 512 # sure both images have the same dimensions
convert_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((image_size, image_size))
])

fire = convert_tensor(fire).view(1, 3, image_size, image_size).to(device)
lion = convert_tensor(lion).view(1, 3, image_size, image_size).to(device)

# initializing the input image to the content image
gen_image = lion.clone()
gen_image = gen_image.to(device)

content_weight = 1
style_weight = 1e4
gen_image.requires_grad = True
optimizer = optim.Adam([gen_image], lr=1e-2)

for epoch in range(300):
    content_features_target = content_feature_extractor(lion)
    style_features_target = style_feature_extractor(fire)
    # what we actually need from the feature maps of the style image is its gram matrices
    gram_matrices_target = {}
    for layer in style_features_target.keys():
        gram_matrices_target[layer] = get_gram_matrix(style_features_target[layer])
    # possibly refactor above code using dictionary comprehension
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
    print(f"epoch {epoch} - content loss: {l_content} || style loss: {l_style}")
    loss = style_weight * l_style + content_weight * l_content
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


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
display_image(fire)

plt.figure()
display_image(gen_image)