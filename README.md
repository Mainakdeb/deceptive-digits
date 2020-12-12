# deceptive-digits
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mainakdeb/deceptive-digits/blob/main/deceptive-digits.ipynb?authuser=1)

Use conditional-dcgans to generate realistic images of digits.

<img src="https://github.com/Mainakdeb/deceptive-digits/blob/main/images/mnist_cgan_demo.gif" width="610">

The digits you see above are not real, they are generated using a PyTorch based Conditional DCGAN, but what if you wanted to generate images of numbers with multiple digits? I've used numpy to crop, invert and stich generated images horizontally, check out the generated image below, the input was 9830976450, and each individual digit was cropped, inverted and stiched together with low spacing to improve realism.

<img src="https://github.com/Mainakdeb/deceptive-digits/blob/main/images/generated9830976450%20_with_title.png" width="610">

I've prepared a colab notebook, feel free to check it out.

## What Exactly is a Conditional GAN ? 
The **condition** in a Conditional GAN is basically the desired class of the generated sample, it is a generative model that allows the generation of targeted data from a given class. Unlike a traditional GAN, it also accounts for the labels while training. 


## Using the labels:
I was extremely confused on how the labels were to be used. But then I found the following diagram in the [Conditional-GAN paper](https://arxiv.org/abs/1411.1784)

<img src="https://github.com/Mainakdeb/deceptive-digits/blob/main/images/cgan_paper_diagram.png" width="610">

  * x represents real-life data.
  * z represents random noise.
  * y represents the corresponding label.
  * D(x|y) and G(z|y) represent the discriminator net and the generator net respectively.
## What I did:

I used Conv layers in both the generator and the discriminator. Here's how I used the labels:

  * **The Generator**: Take a look at the diagram below
  <img src="https://github.com/Mainakdeb/deceptive-digits/blob/main/images/mnist_cgan_generator.png" width="1200">

  I concatenated a random noise vector of length 100 with another vector that represents a label of length 10 and passed the resulting tensor through the generator net. Notice that the input tensor has a length of 110. This input vector is passed through transpose-convolution layers to generate a 1x64x64 image.

  * **The Discriminator**: Take a look at the diagram below
  <img src="https://github.com/Mainakdeb/deceptive-digits/blob/main/images/mnist_cgan_discriminator.png" width="1200">
  A 1x64x64 image is passed through convolution layers and the resulting tensor of length 10 is concatenated with a label embedding of length 10 and the resulting tensor is passed through the linear layers.  Notice that the first linear layer takes an input of length 20. The final output is a tensor of length 1, which represents the probability of the sample being real or fake.

## Inference:
I kept an eye on the losses of the models and plotted generated samples after every epoch. After about 7 epochs, the generator was able to generate reasonably good samples. Each frame in the gif below showcases the generators performance through the epochs.

<img src="https://github.com/Mainakdeb/deceptive-digits/blob/main/images/mnist_cgan_training_timelapse_compressed.gif" width="1200">

Notice how it starts off with alien looking digits, and keeps improving over time. The only form of augmentation I used is random rotation. I believe using more advanced augmentation techniques will improve the overall performance.

## Exploring the Latent Space:
The generator net (here) accepts a latent vector of length 100 and a label embeddding of length 10. While the network trains, it learns to map these latent points to generated images. Every single latent vector is a point in an n-dimensional space where n is the length of the latent vector, which is 100 in this case. 

What if you take 2 points in this 100-dimensional space and generate samples by interpolating between them? Every adjacent point leads to the generation of a slightly different image. The following gif showcases generated images with latent vectors interpolated between 2 points, looping back and forth between the 2 extremes.

<img src="https://github.com/Mainakdeb/deceptive-digits/blob/main/images/mnist_cgan_interps_loop_10_compressed.gif" width="1200">


Notice how each frame is slightly different from the previous. Lets visualize some interpolations side by side, with a different set of interpolated points:

<img src="https://github.com/Mainakdeb/deceptive-digits/blob/main/images/mnist_cgan_all_interps.png" width="1200">

## Credits:
1. The [Conditional-GAN paper](https://arxiv.org/abs/1411.1784)
2. For network visualizations - [NN-SVG](https://alexlenail.me/NN-SVG/)
3. For creating gifs - [Ezgif](https://ezgif.com/)

