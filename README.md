# [MOREH] Running on the HAC VM - Moreh AI Framework
## 1. Clone, prepare environment:
```bash
git clone https://github.com/moreh-dev/ml-workbench.git
cd ml-workbench/DCGAN-PyTorch
```
## Note: some changes have been made compared to original repo to fix the issue and make the training run well in our MAF:
* In train.py, line 70-71, added float() outside of the int value to avoid Float issue.
```bash
real_label = float(1)
fake_label = float(0)
```
* In train.py, bottom line 204, changed writer from imagemagick to pillow:
```bash
anim.save('celeba.gif', dpi=80, writer='pillow')
```
## 2. Prepare data by manually download as guided or by other method, then update the directory location inside the `root` variable in **`utils.py`**(line 6).
- This implementation uses the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset. However, any other dataset can also be used. Download from https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg and preprocess if need.
- Other ways: https://gist.github.com/SeitaroShinagawa/05083f971d3f1b88a19df32841e5cb25 or https://github.com/suvojit-0x55aa/celebA-HQ-dataset-download

## 2b. Install pillow to write gif:
```bash
pip install pillow
```

## 4. Start training: 
To train the model, run **`train.py`**. To set the training parametrs, update the values in the `params` dictionary in `train.py`.
Checkpoints would be saved by default in model directory every 2 epochs.
```bash
python train.py
```
------------

# Original README:
# Deep Convolutional GAN

PyTorch implementation of DCGAN introduced in the paper: [Unsupervised Representation Learning with Deep Convolutional 
Generative Adversarial Networks](https://arxiv.org/abs/1511.06434), Alec Radford, Luke Metz, Soumith Chintala.

<p align="center">
<img src="gen_celeba.gif" title="Generated Data Animation" alt="Generated Data Animation">
</p>

## Introduction
Generative Adversarial Networks (GANs) are one of the most popular (and coolest)
Machine Learning algorithms developed in recent times. They belong to a set of algorithms called generative models, which
are widely used for unupervised learning tasks which aim to learn the uderlying structure of the given data. As the name
suggests GANs allow you to generate new unseen data that mimic the actual given real data. However, GANs pose problems in
training and require carefullly tuned hyperparameters.This paper aims to solve this problem.

DCGAN is one of the most popular and succesful network design for GAN. It mainly composes of convolution layers 
without max pooling or fully connected layers. It uses strided convolutions and transposed convolutions 
for the downsampling and the upsampling respectively.

**Generator architecture of DCGAN**
<p align="center">
<img src="images/Generator.png" title="DCGAN Generator" alt="DCGAN Generator">
</p>

**Network Design of DCGAN:**
* Replace all pooling layers with strided convolutions.
* Remove all fully connected layers.
* Use transposed convolutions for upsampling.
* Use Batch Normalization after every layer except after the output layer of the generator and the input layer of the discriminator.
* Use ReLU non-linearity for each layer in the generator except for output layer use tanh.
* Use Leaky-ReLU non-linearity for each layer of the disciminator excpet for output layer use sigmoid.

## Hyperparameters for this Implementation
Hyperparameters are chosen as given in the paper.
* mini-batch size: 128
* learning rate: 0.0002
* momentum term beta1: 0.5
* slope of leak of LeakyReLU: 0.2
* For the optimizer Adam (with beta2 = 0.999) has been used instead of SGD as described in the paper.

## Data
This implementation uses the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset. However, any other dataset can
also be used. Download the data and update the directory location inside the `root` variable in **`utils.py`**.

**CelebA Dataset**
<p align="center">
<img src="images/Training_Data.png" title="Training Data" alt="Training Data">
</p>

## Training
To train the model, run **`train.py`**. To set the training parametrs, update the values in the `params` dictionary in `train.py`.
Checkpoints would be saved by default in model directory every 2 epochs. 
By default, GPU is used for training if available.

*Training will take a long time. It took me around 3 hours on a NVIDIA GeForce GTX 1060 GPU. Using a CPU is not recommended.*

**Loss Curves**
<p align="center">
<img src="images/Training_Loss.png" title="Training Loss Curves" alt="Training Loss Curves">
</p>
<i>D: Discriminator, G: Generator</i>

## Generating New Images
To generate new unseen images, run **`generate.py`**.
```sh
python3 generate.py --load_path /path/to/pth/checkpoint --num_output n
```
**Generated Images**
<p align="center">
  <i>After Epoch 1:</i> <img src="images/Generated_Epoch_1.png" title="Generated Images after 1st Epoch" alt="Generated Images after 1st Epoch">
  <i>After Epoch 10:</i> <img src="images/Generated_Epoch_10.png" title="Generated Images after 10th Epoch" alt="Generated Images after 10th Epoch">
</p>

## References
1. **Alec Radford, Luke Metz, Soumith Chintala.** *Unsupervised representation learning with deep convolutional 
generative adversarial networks.*[[arxiv](https://arxiv.org/abs/1511.06434)]
2. **Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, 
Sherjil Ozair, Aaron Courville, Yoshua Bengio.** *Generative adversarial nets.* NIPS 2014 [[arxiv](https://arxiv.org/abs/1406.2661)]
3. **Ian Goodfellow.** *Tutorial: Generative Adversarial Networks.* NIPS 2016 [[arxiv](https://arxiv.org/abs/1701.00160)]
4. DCGAN Tutorial. [https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html]
5. PyTorch Docs. [https://pytorch.org/docs/stable/index.html]
