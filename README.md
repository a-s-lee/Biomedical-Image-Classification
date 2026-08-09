# Biomedical Image Classification & Analysis

A deep learning project using **PyTorch** to classify biomedical blood cell images from the **BloodMNIST** dataset into 8 cell types. The project compares optimization strategies, evaluates classification performance, and analyzes model behavior through training curves and confusion matrices.

## Project Overview

This project explores the use of a **Multi-Layer Perceptron (MLP)** for biomedical image classification. The model was trained and evaluated on more than **17,000 blood cell images** across 8 classes.

The project focuses on:

- Preparing biomedical image data for deep learning
- Building and training a PyTorch neural network
- Comparing SGD and Adam optimizers
- Tracking experiments with Weights & Biases
- Evaluating model performance across multiple blood cell classes
- Visualizing training behavior and classification errors

## Dataset

This project uses the **BloodMNIST** dataset from MedMNIST, containing
17,092 blood cell images across 8 classes.

| Split | Images |
|---|---:|
| Training | 11,959 |
| Validation | 1,712 |
| Testing | 3,421 |
| **Total** | **17,092** |

The dataset is **not included** in this repository due to file size.
It can be obtained through the MedMNIST dataset package.

## Technologies

- **Python**
- **PyTorch**
- **NumPy**
- **Matplotlib**
- **scikit-learn**
- **Weights & Biases**
- **Jupyter Notebook**

## Data Preparation

The BloodMNIST images were converted into PyTorch tensors and normalized from pixel values of 0–255 to values between 0 and 1.

The data was then organized into training, validation, and testing DataLoaders with a batch size of **32**.

A fixed random seed was also used to improve experiment reproducibility.

## Model Architecture

The project uses a fully connected **Multi-Layer Perceptron** with three hidden layers:

```text
Input: 28 × 28 × 3
        ↓
512 neurons
        ↓
256 neurons
        ↓
128 neurons
        ↓
8 output classes
```

The network incorporates:

- ReLU activation
- Batch normalization
- 50% dropout
- Cross-entropy loss
- Weight decay regularization

These techniques were used to improve training stability and reduce overfitting.

## Model Training

Two optimization strategies were compared:

### Stochastic Gradient Descent
- Momentum: 0.9
- Learning rate: 0.001
- Weight decay: 0.0001

### Adam
- Learning rate: 0.001
- Weight decay: 0.0001

Both models were trained for **50 epochs**.

Training and validation accuracy and loss were tracked using **Weights & Biases**, while the model with the highest validation accuracy was retained for evaluation.

## Results

| Optimizer | Best Validation Accuracy |
|---|---:|
| **SGD** | **88.79%** |
| Adam | 88.08% |

**SGD achieved the highest validation accuracy at 88.79%.**

Despite Adam learning faster during the early stages of training, SGD ultimately produced the strongest validation performance.

## Model Evaluation

Model performance was evaluated using:

- Training accuracy
- Validation accuracy
- Test accuracy
- Training and validation loss
- Confusion matrices
- Class-level prediction errors

Confusion matrices were generated for both optimizers to identify which blood cell classes were most frequently confused by the model.

## Key Takeaways

- Trained an MLP on **17,092 biomedical images across 8 classes**
- Achieved **88.79% best validation accuracy**
- Compared **SGD and Adam** across 50 training epochs
- Used **batch normalization and dropout** to improve generalization
- Tracked model experiments using **Weights & Biases**
- Analyzed class-level performance using **confusion matrices**

## Repository Structure

```text
Biomedical-Image-Classification/
│
├── Biomedical Classification.ipynb
├── data/
│   └── bloodmnist.npz
└── README.md
```

## Future Improvements

Potential improvements include:

- Replacing the MLP with a Convolutional Neural Network (CNN)
- Applying image augmentation
- Performing additional hyperparameter tuning
- Comparing additional optimizers and learning rates
- Evaluating precision, recall, and F1-score by class

## Author

**Alex Lee**

Data Science | Data Analytics | Machine Learning
