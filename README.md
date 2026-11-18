# Clothing Identifier ML

This project compares two machine-learning approaches for classifying images
from the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
dataset:

- A custom, batched k-nearest neighbors (k-NN) classifier
- A scikit-learn multilayer perceptron (MLP) neural network

The notebook downloads the dataset from OpenML, splits it into training,
validation, and test sets, scales image pixels to `[-1, 1]`, searches model
hyperparameters, and reports accuracy.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Start Jupyter and open [`knn-nn.ipynb`](./knn-nn.ipynb):

```bash
jupyter notebook
```

Run the cells from top to bottom. The first run downloads Fashion-MNIST from
OpenML, so an internet connection is required.

## Results

The recorded test accuracies are:

| Model | Test accuracy |
| --- | ---: |
| k-NN | 85.15% |
| MLP neural network | 89.65% |

Generated visualizations are available in [`fig/`](./fig/), including the k-NN
grid-search curve and the neural-network loss curve.
