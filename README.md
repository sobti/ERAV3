# MNIST AI Model

This project trains a Convolutional Neural Network (CNN) to classify handwritten digits (0-9) from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). The model is implemented in PyTorch and uses transformations to augment the training data for improved accuracy.

## Features

- **Data Augmentation**: Includes brightness, contrast, rotation, and normalization.
- **Model Architecture**: A simple CNN for digit classification.
- **Training Pipeline**: Automated training and testing with accuracy reporting.
- **Artifact Uploads**: Saves transformed images and trained model files.
- **GitHub Actions Integration**: CI/CD pipeline for reproducible training.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/mnist-ai-model.git
   cd mnist-ai-model
2. Install dependencies: Make sure you have Python 3.8 or higher installed. Then install the required libraries:
3. Download MNIST Dataset: The dataset will automatically be downloaded during the first training run.
Usage

Train the Model
Run the training script:

python train_model.py
This will:

Train the CNN on the MNIST training dataset.
Evaluate the model on the test dataset.
Save the trained model to a timestamped file in the project directory.
View Transformed Images
The script saves transformed training images to the transformed_images/ directory. You can view these images to verify the preprocessing pipeline.

GitHub Actions
This repository includes a GitHub Actions workflow to:

Train the model in a CI environment.
Save transformed images and the trained model as artifacts
