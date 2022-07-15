# TODO: Make a video rendering functionality
# TODO: Consider normalizing the weights to add up to 1
# TODO: Try to make the program work better
#       for when starting from noise

import argparse
from utils import prepare_image, save_img, VGG_MEAN
import torch
from models import VGG19
import torch.optim as optim

# consider changing this to be chosen explicitly by user
device = {torch.has_cuda: "cuda",
          torch.has_mps: "mps"}.get(True, "cpu")


def generate_image(args):

    print(f"Using {device} device")

    # loading test images
    content_img = prepare_image(args.content_image, args.image_size).to(device)
    style_img = prepare_image(args.style_image, args.image_size).to(device)

    if args.normalize_weights:

        total_weight = args.content_weight + args.style_weight + args.smoothness
        args.content_weight = args.content_weight / total_weight
        args.style_weight = args.style_weight / total_weight
        args.smoothness = args.smoothness / total_weight

    # the forward call to this model returns the losses with respect these images
    model = VGG19(content_img, style_img, args.content_weight,
                  args.style_weight, args.smoothness,
                  args.pooling, args.content_layers, args.style_layers, device)

    # initializing the input image to the content image
    init_image_method = args.init
    if init_image_method == "content":
        gen_image = content_img.clone()
    elif init_image_method == "style":
        gen_image = style_img.clone()
    elif init_image_method == "noise":
        gen_image = torch.randn(content_img.shape).mul(1e-3).add(0.5)
        gen_image -= VGG_MEAN.view(1, 3, 1, 1)
        gen_image = gen_image.to(device)

    # consider experimenting with Adam
    # setting up for optimization
    gen_image.requires_grad = True
    optimizer = optim.LBFGS([gen_image], max_iter=args.iter,
                            tolerance_change=-1, tolerance_grad=-1)

    step_cnt = [0]
    # generating the target image
    while step_cnt[0] < 1:

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
                if step_cnt[0] % args.disp_iter == 0:
                    print(
                        f"step {step_cnt[0]} \tcontent loss: {content_loss} \tstyle loss: {style_loss}")

            loss = content_loss + style_loss + tv_loss
            loss.backward()

            step_cnt[0] += 1
            return loss

        optimizer.step(closure)

    save_img(gen_image, args.save_path)


def main():
    parser = argparse.ArgumentParser(
        description="parser for artistic neural net")

    # consider specifying the default values in help
    parser.add_argument("--content-image", required=True,
                        help="path of the content image")
    parser.add_argument("--style-image", required=True,
                        help="path of the style image")
    parser.add_argument("--save-path", default="untitled.png",
                        help="name and path where generated image will be saved")
    parser.add_argument("--image-size", type=int, default=256,
                        help="the size of the generated image")
    parser.add_argument("--content-weight", type=float, default=1,
                        help="style loss weight")
    parser.add_argument("--style-weight", type=float, default=1e5,
                        help="style loss weight")
    parser.add_argument("--smoothness", type=float, default=1e-1,
                        help="total variation loss weight to make image smoother")
    # consider changing this to a more useful name
    parser.add_argument("--init", default="content",
                        help="initial image to be used",
                        choices=["content", "noise"])
    parser.add_argument("--pooling", default="max",
                        help="the pooling used in the network",
                        choices=["max", "avg"])
    parser.add_argument("--iter", type=int, default=500,
                        help="number of optimization steps")
    parser.add_argument("--disp-iter", type=int, default=50,
                        help="number of optimization steps before error is displayed")
    parser.add_argument("--content-layers", nargs="+",
                        default=["relu4_2"],
                        help="specify the content layers, space separated")
    parser.add_argument("--style-layers", nargs="+",
                        default=["relu1_1", "relu2_1",
                                 "relu3_1", "relu4_1", "relu5_1"],
                        help="specify the style layers, space separated")
    parser.add_argument("--normalize-weights", action="store_true",
                        help="include to normalize weights to add up to 1")
    args = parser.parse_args()
    generate_image(args)


if __name__ == "__main__":
    main()
