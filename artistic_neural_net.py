# TODO: Make a video rendering functionality after cleaning up code
# TODO: Perhaps allow configuration through a dictionary passed to the main function
# TODO: Replace max pooling with average pooling with perhaps
#       trying to have more control over feature extraction
#       (writing out the entire VGG19 network on a separate file)
# TODO: Implement total viariation loss to improve smoothness

from utils import *
import torch.optim as optim
from torchvision.models import vgg19, VGG19_Weights
from torchvision.models.feature_extraction import create_feature_extractor


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
    content_img = prepare_image("images/houses.jpg", 256).to(device)
    style_img = prepare_image("images/night.jpg", 256).to(device)

    # initializing the input image to the content image
    # randn works better than just rand for when starting from random noise?
    gen_image = content_img.clone()

    content_features_target, gram_matrices_target = get_features(
        content_img, style_img, detach=True)

    # setting up for optimization
    content_weight = 1
    style_weight = 1e3
    gen_image.requires_grad = True
    optimizer = optim.LBFGS([gen_image])

    step_cnt = [0]
    # generating the target image
    while step_cnt[0] <= 201:

        def closure():
            optimizer.zero_grad()
            gen_features_content, gram_matrices_gen = get_features(
                gen_image, gen_image)

            # getting the total loss
            l_style = 0
            l_content = 0
            # for now, I will let all considered layers contribute equally
            for layer in gram_matrices_target:
                l_style += nn.functional.mse_loss(gram_matrices_gen[layer],
                                                  gram_matrices_target[layer], reduction="mean")

            for layer in gen_features_content:
                l_content += nn.functional.mse_loss(
                    gen_features_content[layer], content_features_target[layer], reduction="mean")

            with torch.no_grad():
                if step_cnt[0] % 50 == 0:
                    print(
                        f"step {step_cnt[0]} \tcontent loss: {content_weight * l_content} \tstyle loss: {style_weight * l_style}")

            loss = style_weight * l_style + content_weight * l_content
            loss.backward()

            step_cnt[0] += 1
            return loss

        optimizer.step(closure)

    save_img(gen_image)


if __name__ == "__main__":
    artistic_neural_net()
