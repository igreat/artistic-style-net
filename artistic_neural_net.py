# TODO: Make a video rendering functionality after cleaning up code
# TODO: Perhaps allow configuration through a dictionary passed to the main function
# TODO: Implement total viariation loss to improve smoothness

from utils import *
from models import VGG19
import torch.optim as optim


def artistic_neural_net():

    device = {torch.has_cuda: "cuda",
              torch.has_mps: "mps"}.get(True, "cpu")

    print(f"Using {device} device")

    # loading test images
    content_img = prepare_image("images/houses.jpg", 256).to(device)
    style_img = prepare_image("images/night.jpg", 256).to(device)

    # the forward call to this model returns the losses with respect these images
    model = VGG19(content_img, style_img, device=device)

    # initializing the input image to the content image
    # randn works better than just rand for when starting from random noise?
    init_image_method = "content"
    if init_image_method == "content":
        gen_image = content_img.clone()
    elif init_image_method == "style":
        gen_image = style_img.clone()
    else:  # defaults to a white noise
        gen_image = transforms.Resize(content_img.shape[2:])(
            prepare_image("images/white-noise.jpg").to(device))

    # setting up for optimization
    content_weight = 1
    style_weight = 1e3
    gen_image.requires_grad = True
    optimizer = optim.LBFGS([gen_image])

    step_cnt = [0]
    # generating the target image
    while step_cnt[0] <= 250:

        def closure():
            optimizer.zero_grad()

            content_loss, style_loss = model(gen_image)

            content_loss *= content_weight
            style_loss *= style_weight

            with torch.no_grad():
                if step_cnt[0] % 50 == 0:
                    print(
                        f"step {step_cnt[0]} \tcontent loss: {content_loss} \tstyle loss: {style_loss}")

            loss = content_loss + style_loss
            loss.backward()

            step_cnt[0] += 1
            return loss

        optimizer.step(closure)

    save_img(gen_image)


if __name__ == "__main__":
    artistic_neural_net()
