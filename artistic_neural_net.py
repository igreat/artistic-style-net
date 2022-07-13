# TODO: Make a video rendering functionality after cleaning up code
# TODO: Implement argument parsing for customizable user input
# TODO: Implement total viariation loss to improve smoothness

from utils import *
from models import VGG19
import torch.optim as optim


def main():

    device = {torch.has_cuda: "cuda",
              torch.has_mps: "mps"}.get(True, "cpu")

    print(f"Using {device} device")

    # loading test images
    content_img = prepare_image(
        "images/sultan-qaboos-grand-mosque.jpg", 256).to(device)
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
    else:  # defaults to noise
        gen_image = torch.randn(content_img.shape).mul(0.001).to(device)

    # setting up for optimization
    gen_image.requires_grad = True
    optimizer = optim.LBFGS([gen_image], line_search_fn="strong_wolfe")

    step_cnt = [0]
    # generating the target image
    while step_cnt[0] <= 300:

        def closure():
            optimizer.zero_grad()

            content_losses, style_losses, tv_loss = model(gen_image)

            content_loss = 0
            style_loss = 0
            for loss in content_losses:
                content_loss += loss
            for loss in style_losses:
                style_loss += loss

            with torch.no_grad():
                if step_cnt[0] % 50 == 0:
                    print(
                        f"step {step_cnt[0]} \tcontent loss: {content_loss} \tstyle loss: {style_loss}")

            loss = content_loss + style_loss + tv_loss
            loss.backward()

            step_cnt[0] += 1
            return loss

        optimizer.step(closure)

    save_img(gen_image, "generated images/test.png")


if __name__ == "__main__":
    main()
