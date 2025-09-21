# Generative Adversarial Network (GAN) Implementation in PyTorch

This repository contains a Jupyter Notebook for a deep learning assignment focused on implementing Generative Adversarial Networks (GANs) from scratch using PyTorch. The project provides a modular `GANTrainer` class and demonstrates the GAN's capabilities on two distinct tasks: learning the distribution of 2D toy data and generating handwritten digits from the MNIST dataset.

## Key Features

- **Modular Trainer Class:** A reusable `GANTrainer` class that handles the training loop for both the generator and the discriminator.
- **Two Distinct Examples:**
    1.  **2D Toy Dataset:** Visualizes the GAN's ability to learn a complex data distribution (a grid of Gaussians) and provides a clear example of the **Mode Collapse** problem.
    2.  **MNIST Dataset:** A practical application of generating realistic grayscale images of handwritten digits.
- **Core GAN Concepts:** Implements fundamental GAN components, including generator/discriminator architectures, different loss functions (`log(D)` vs. `log(1-D)`), and the alternating training process.
- **Visualization:** Includes helper functions to visualize the generator's output distribution, the discriminator's decision boundary, and interpolations in the latent space.

## Project Structure

The entire implementation is contained within a single Jupyter Notebook:

-   `GAN.ipynb`: The notebook containing all the code, explanations, and problem solutions for the assignment.

## Datasets

This project utilizes two datasets to explore GANs:

1.  **Toy2dGridGaussiansDataset:** A custom-built 2D dataset consisting of a grid of Gaussian distributions. It is used to visually inspect the learning process and diagnose issues like mode collapse.
2.  **MNIST:** The standard dataset of 28x28 grayscale images of handwritten digits, used for the image generation task. The notebook will automatically download it.

## Concepts Covered

-   Generative Adversarial Networks (Generator & Discriminator)
-   GAN Loss Functions
-   Deep Convolutional GAN (DCGAN) architectural principles
-   Training Dynamics (alternating updates for G and D)
-   Mode Collapse
-   Latent Space Interpolation
