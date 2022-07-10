# TODO: Make a video rendering functionality after cleaning up code
# TODO: Perhaps allow configuration through a dictionary passed to the main function

from utils import *
import torch.optim as optim
from torchvision.models import vgg19, VGG19_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import torch


def artistic_neural_net():

    device = {torch.has_cuda: "cuda",
              torch.has_mps: "mps"}.get(True, "cpu")

    print(f"Using {device} device")

    # # loading pretrained model
    model = vgg19(weights=VGG19_Weights.DEFAULT).eval().to(device)

    model.requires_grad = False

    # # setting up feature extractors
    style_feature_extractor = create_feature_extractor(
        model, return_nodes=STYLE_NODES_RELU).to(device)

    content_feature_extractor = create_feature_extractor(
        model, return_nodes=CONTENT_NODES).to(device)

    get_features = get_content_and_style(
        content_feature_extractor, style_feature_extractor)
    # loading test images
    lion = prepare_image("images/lion.jpg", 256).to(device)
    night = prepare_image("images/night.jpg", 256).to(device)

    # initializing the input image to the content image
    gen_image = lion.clone()

    content_features_target, gram_matrices_target = get_features(
        lion, night, detach=True)

    # setting up for optimization
    content_weight = 1
    style_weight = 1e5
    gen_image.requires_grad = True
    optimizer = optim.LBFGS([gen_image], max_iter=251)

    global step_cnt
    step_cnt = 0

    # generating the target image
    def closure():
        global step_cnt
        optimizer.zero_grad()
        gen_features_content, gram_matrices_gen = get_features(
            gen_image, gen_image)

        # getting the total loss
        l_style = 0
        l_content = 0
        # for now, I will let all considered layers contribute equally
        for layer in gram_matrices_target.keys():
            l_style += style_loss(gram_matrices_gen[layer],
                                  gram_matrices_target[layer])

        for layer in gen_features_content.keys():
            l_content += content_loss(
                gen_features_content[layer], content_features_target[layer])

        with torch.no_grad():
            if step_cnt % 50 == 0:
                print(
                    f"step {step_cnt} \tcontent loss: {content_weight * l_content} \tstyle loss: {style_weight * l_style}")

        loss = style_weight * l_style + content_weight * l_content
        loss.backward()

        step_cnt += 1
        return loss

    optimizer.step(closure)

    save_img(gen_image)


if __name__ == "__main__":
    artistic_neural_net()
