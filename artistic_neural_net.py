import argparse
import utils
import torch
import numpy as np
import torch.optim as optim
from models import VGG19
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms import Resize
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter


# consider changing this to be chosen explicitly by user
device = {torch.has_cuda: "cuda", torch.has_mps: "mps"}.get(True, "cpu")
print(f"Using {device} device")


def generate_image(args):
    summary = SummaryWriter()

    content_img = pil_to_tensor(Image.open(args.content_image)).div(255.0)
    style_img = pil_to_tensor(Image.open(args.style_image)).div(255.0)

    if args.maintain_color:
        content_yiq = utils.rgb_to_yiq(content_img)
        content_iq = Resize(args.image_size)(content_yiq[1:3])
        # must convert back to rgb because that is what model expects
        content_img = utils.gs_to_rgb(content_yiq[0])
        style_img = utils.gs_to_rgb(utils.rgb_to_yiq(style_img)[0])

    # loading test images
    content_img = utils.process_image(content_img, args.image_size).to(device)
    style_img = utils.process_image(style_img, args.image_size).to(device)

    # the forward call to this model returns the losses with respect these images
    model = VGG19(
        content_img,
        style_img,
        args.content_weight,
        args.style_weight,
        args.smoothness,
        args.pooling,
        args.content_layers,
        args.style_layers,
        device,
    )

    # initializing the input image to the content image
    init_image_method = args.init
    if args.init_image:
        # not yet supported for no color transfer
        gen_image = pil_to_tensor(Image.open(args.init_image)).div(255.0)
        gen_image = utils.process_image(gen_image, args.image_size).to(device)
    elif init_image_method == "content":
        gen_image = content_img.clone()
    elif init_image_method == "style":
        # doesn't currently work because content error is spatially dependent
        gen_image = style_img.clone()
    elif init_image_method == "noise":
        gen_image = torch.randn_like(content_img)
        gen_image = gen_image.to(device)

    # consider experimenting with Adam
    # setting up for optimization
    gen_image.requires_grad = True
    optimizer = optim.LBFGS(
        [gen_image], max_iter=args.iter, tolerance_change=-1, tolerance_grad=-1
    )

    step_cnt = [0]
    # generating the target image
    while step_cnt[0] < 1:

        def optim_step():

            optimizer.zero_grad()

            content_losses, style_losses, tv_loss = model(gen_image)

            content_loss = 0
            style_loss = 0
            for loss in content_losses:
                content_loss += loss
            for loss in style_losses:
                style_loss += loss

            if step_cnt[0] % args.disp_iter == 0:
                print(
                    f"step {step_cnt[0]} \tcontent loss: {content_loss} \tstyle loss: {style_loss}"
                )

                # preparing and displaying the styled image
                # deprocess the image
                result = utils.deprocess_image(gen_image.detach().clone())
                result = result.clamp(0, 1) * 255
                result = result.cpu().numpy().astype(np.uint8).squeeze(0)
                summary.add_image(
                    "styled_image",
                    result,
                    step_cnt[0],
                )

            loss = content_loss + style_loss + tv_loss
            loss.backward()

            # adding losses to tensorboard
            summary.add_scalar(
                "losses/content",
                content_loss.item(),
                step_cnt[0],
            )
            summary.add_scalar(
                "losses/style",
                style_loss.item(),
                step_cnt[0],
            )
            summary.add_scalar(
                "losses/tv",
                tv_loss.item(),
                step_cnt[0],
            )

            step_cnt[0] += 1
            return loss

        optimizer.step(optim_step)

    gen_image = utils.deprocess_image(gen_image)

    if args.maintain_color:
        gen_image = gen_image.squeeze(0)[0].unsqueeze(0)
        gen_image = torch.cat((gen_image.to("cpu"), content_iq), 0)
        gen_image = utils.yiq_to_rgb(gen_image)

    return gen_image


def main():
    parser = argparse.ArgumentParser(description="parser for artistic neural net")

    # consider specifying the default values in help
    parser.add_argument(
        "--content-image", required=True, help="path of the content image"
    )
    parser.add_argument("--style-image", required=True, help="path of the style image")
    parser.add_argument(
        "--save-path",
        default="untitled.png",
        help="name and path where generated image will be saved",
    )
    parser.add_argument(
        "--image-size", type=int, default=256, help="the size of the generated image"
    )
    parser.add_argument(
        "--content-weight", type=float, default=1, help="style loss weight"
    )
    parser.add_argument(
        "--style-weight", type=float, default=1e4, help="style loss weight"
    )
    parser.add_argument(
        "--smoothness",
        type=float,
        default=1e-4,
        help="total variation loss weight to make image smoother",
    )
    # consider changing this to a more useful name
    parser.add_argument(
        "--init",
        default="content",
        help="initial image to be used",
        choices=["content", "noise", "image"],
    )
    parser.add_argument(
        "--init-image", default=None, help="specify path to initial image"
    )
    parser.add_argument(
        "--maintain-color",
        action="store_true",
        help="include to maintain the original color of the content image",
    )
    parser.add_argument(
        "--pooling",
        default="max",
        help="the pooling used in the network",
        choices=["max", "avg"],
    )
    parser.add_argument(
        "--iter", type=int, default=500, help="number of optimization steps"
    )
    parser.add_argument(
        "--disp-iter",
        type=int,
        default=50,
        help="number of optimization steps before error is displayed",
    )
    parser.add_argument(
        "--content-layers",
        nargs="+",
        default=["relu4_2"],
        help="specify the content layers, space separated",
    )
    parser.add_argument(
        "--style-layers",
        nargs="+",
        default=["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"],
        help="specify the style layers, space separated",
    )
    args = parser.parse_args()

    image = generate_image(args)
    save_image(image, args.save_path)


if __name__ == "__main__":
    main()
