# NumPy neural network for MNIST

A handwritten digit classifier built with NumPy, including forward propagation,
ReLU hidden layers, softmax outputs, and backpropagation implemented from scratch.

## Setup and run

Requires Python 3.10 or newer.

```sh
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies and run:

```sh
python -m pip install -r requirements.txt
python mnist_learning.py
```

The first run downloads MNIST into `data/mnist/` from the
[CVDF MNIST mirror](https://github.com/cvdfoundation/mnist).
Pixel brightness is normalized to 0–1.

The current example trains a `784 → 128 → 64 → 10` network for 19 epochs
at learning rate `0.003`. It prints training accuracy, evaluates test accuracy,
and displays nine incorrectly classified test images.

## Files and features

- `mnist.py`: dataset downloading, loading, normalization, and sample display.
- `mnist_learning.py`: configurable networks, training, evaluation, validation
  splitting, architecture/learning-rate search, early stopping, and mistake plots.
- `ml.py`: earlier Iris classification experiments.

The architecture search example near the bottom of `mnist_learning.py` is
currently disabled inside a multiline string. Enable it instead of the single
network example to compare configurations. Select settings with validation
accuracy and evaluate the test set after selection.

Search saves each new best model to `best_network.npz`. This saves the selected
parameters, not the state needed to resume an interrupted search.

Downloaded datasets, model checkpoints, and Python caches are excluded from Git.
Training currently processes one image at a time on the CPU; large searches can
take considerable time. Importing `mnist_learning.py` currently also runs its
example, so run it as a script.
