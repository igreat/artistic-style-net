# Artistic Nueral Transfer 🖌️🎨🧠

## Background

In this repository I will do a PyTorch implementation of the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge.

By using a deep CNN trained on a large dataset, we can extract **content** features and **style** features in order to produce the desired combination of the two in one image. Here is an example of the Sultan Qaboos Grand Mosque mixed with the style of the Starry Night by Vincent van Gogh.

<div align="center">
    <img src="images/test-images/starry-night.jpg" alt="Starry Night" width="256"/>
    <img src="images/test-images/sultan-qaboos-grand-mosque.jpg" alt="Sultan Qaboos Grand Mosque" width=256/>
    <img src="images/generated-images/night-grand-mosque.png" alt="Starry Grand Mosque" width="512"/>
</div>

## Results

Here is a portrait of an apple put with different style images.

<div align="center">
    <div>
        <img src="images/apples/apple.jpg" width="140px"/>
    </div>
    <span>
        <img src="images/test-images/starry-night.jpg" alt="Starry Night" height="150"/>
        <img src="images/apples/starry-apple.png" height="150px"/>
    </span>
    <span>
        <img src="images/test-images/candy.jpg" height="150"/>
        <img src="images/apples/candy-apple.png" height="150px"/>
    </span>
    <span>
        <img src="images/test-images/picasso.jpg" height="150"/>
        <img src="images/apples/picasso-apple.png" height="150px"/>
    </span>
    <span>
        <img src="images/test-images/the-scream.jpg" alt="Starry Night" height="150"/>
        <img src="images/apples/scream-apple.png" height="150px"/>
    </span>
</div>

By using different weights on the style and content images, we can get different results as shown below where the ratios between the weights of the style image and the content image are 1e4, 1e5, 1e6, 1e7 respectively from left to right.

<div align="center">
    <img src="images/weight-comparisons/compare1to1e4.png" width="200px"/>
    <img src="images/weight-comparisons/compare1to1e5.png" width="200px"/>
    <img src="images/weight-comparisons/compare1to1e6.png" width="200px"/>
    <img src="images/weight-comparisons/compare1to1e7.png" width="200px"/>
</div>

## Installation
Soon to be done
## Usage

Here is an example the simplest way to use this program

```
% python artistic_neural_net.py --content-image <PATH TO CONTENT IMAGE> --style-image <PATH TO STYLE IMAGE>
```

The previous command will generate an image and place it in the same directory as the project file, with all the default hyperparamers.

For more customizability, here is a list of all the arguments that can be used.

Soon I will manually provide a usage guide but for now, here is what the help message gives
```
optional arguments:
  -h, --help            show this help message and exit
  --content-image CONTENT_IMAGE
                        path of the content image
  --style-image STYLE_IMAGE
                        path of the style image
  --save-path SAVE_PATH
                        name and path where generated image will be saved
  --image-size IMAGE_SIZE
                        the size of the generated image
  --content-weight CONTENT_WEIGHT
                        style loss weight
  --style-weight STYLE_WEIGHT
                        style loss weight
  --smoothness SMOOTHNESS
                        total variation loss weight to make image smoother
  --init {content,noise}
                        initial image to be used
  --maintain-color      include to maintain the original color of the content image
  --pooling {max,avg}   the pooling used in the network
  --iter ITER           number of optimization steps
  --disp-iter DISP_ITER
                        number of optimization steps before error is displayed
  --content-layers CONTENT_LAYERS [CONTENT_LAYERS ...]
                        specify the content layers, space separated
  --style-layers STYLE_LAYERS [STYLE_LAYERS ...]
                        specify the style layers, space separated
```

