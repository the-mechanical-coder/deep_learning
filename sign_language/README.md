Sign Language Classification ModelThis project implements a neural network using PyTorch to classify sign language letters from the Sign Language MNIST dataset. The model processes 28x28 grayscale images of hand signs representing letters A-Z (26 classes) and predicts the corresponding letter.Table of ContentsRequirements (#requirements)
Dataset (#dataset)
Project Structure (#project-structure)
Usage (#usage)
Model Architecture (#model-architecture)
Training (#training)
Evaluation (#evaluation)
Saving the Model (#saving-the-model)
License (#license)

RequirementsTo run this project, you need the following dependencies:Python 3.x
PyTorch
Pandas
NumPy
Joblib

Install the dependencies using:bash

pip install torch pandas numpy joblib

DatasetThe dataset used is the Sign Language MNIST dataset, which consists of:Training set: sign_mnist_train.csv
Validation set: sign_mnist_valid.csv

Each CSV file contains:A label column with the ground truth (0-25, representing letters A-Z).
784 pixel columns (28x28 grayscale images, flattened).

The data is normalized by dividing pixel values by 255 to scale them to [0, 1].Note: Place the dataset files in a data/ directory relative to the script.Project Structure

project_root/
├── data/
│   ├── sign_mnist_train.csv
│   ├── sign_mnist_valid.csv
├── sign_language.joblib  # Saved model (after training)
├── main.py              # Main script
└── README.md            # This file

UsageEnsure the dataset files are in the data/ directory.
Install the required dependencies (see Requirements (#requirements)).
Run the script:bash

python main.py

The script will:Load and preprocess the dataset.
Train the model for 20 epochs.
Print training and validation loss/accuracy for each epoch.
Save the trained model as sign_language.joblib.

Model ArchitectureThe model is a simple feedforward neural network built with PyTorch's nn.Sequential:Input Layer: Flattens the 28x28 image (784 inputs).
Hidden Layer 1: Linear layer (784 → 512) with ReLU activation.
Hidden Layer 2: Linear layer (512 → 512) with ReLU activation.
Output Layer: Linear layer (512 → 26) for 26 classes (A-Z).

The model is compiled and moved to GPU (if available) for faster computation.TrainingOptimizer: Adam
Loss Function: CrossEntropyLoss
Batch Size: 32
Epochs: 20
Device: GPU (if available) or CPU

The training loop:Processes batches from the training DataLoader.
Computes the loss, backpropagates gradients, and updates weights.
Calculates and prints training loss and accuracy per epoch.
Evaluates the model on the validation set and prints loss and accuracy.

EvaluationThe validation loop:Runs the model in evaluation mode (no gradient computation).
Computes loss and accuracy on the validation set.
Prints results for each epoch.

Accuracy is calculated as the proportion of correct predictions over the total number of samples.

