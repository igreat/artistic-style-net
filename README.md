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
    </div>
    <div align="center">
        <img src="images/test-images/starry-night.jpg" alt="Starry Night" height="150"/>
        <img src="images/apples/starry-apple.png" height="150px"/>
    </div>
    <div align="center">
        <img src="images/test-images/candy.jpg" height="150"/>
        <img src="images/apples/candy-apple.png" height="150px"/>
    <div align="center">
        <img src="images/test-images/picasso.jpg" height="150"/>
        <img src="images/apples/picasso-apple.png" height="150px"/>
    </div align="center">
        <div>
        <img src="images/test-images/the-scream.jpg" alt="Starry Night" height="150"/>
        <img src="images/apples/scream-apple.png" height="150px"/>
    </div>

</div>

By using different weights on the style and content images, we can get different results as shown below where the ratios between the weights of the style image and the content image are 1e4, 1e5, 1e6, 1e7 respectively from left to right.

<div align="center">
    <img src="images/weight-comparisons/compare1to1e4.png" width="160px"/>
    <img src="images/weight-comparisons/compare1to1e5.png" width="160px"/>
    <img src="images/weight-comparisons/compare1to1e6.png" width="160px"/>
    <img src="images/weight-comparisons/compare1to1e7.png" width="160px"/>
</div>

## Installation
