#Import modules
import numpy as np
from mnist import load_mnist
import matplotlib.pyplot as plt

#Load the MNIST dataset
x_train, y_train, x_test, y_test = load_mnist()

#Create a random number generator with a fixed seed for reproducibility
rng = np.random.default_rng(42)

#Define the ReLU activation function
def ReLU(value):
    return np.maximum(0, value)

#Define the softmax function to convert scores into probabilities
def softmax(scores):
    shifted_scores = scores - np.max(scores)
    exp_scores = np.exp(shifted_scores)
    return exp_scores / np.sum(exp_scores)

#Pass an image though the neural network to get the scores and activations for each layer
def forward(pixels, weights, biases):
    activations = [pixels]
    scores = []

    for layer in range(len(weights)):
        layer_scores = activations[-1] @ weights[layer] + biases[layer]
        scores.append(layer_scores)

        if layer == len(weights) - 1:
            # Output layer: keep raw scores.
            activations.append(layer_scores)
        else:
            activations.append(ReLU(layer_scores))

    return scores, activations

#Define the training function for a neural network
def train(training_data, weights, biases, learning_rate, epochs, printAccuracy=False, rng=None):

    x_train, y_train = training_data

    # Copy parameters so the originals aren't modified.
    weights = [w.copy() for w in weights]
    biases = [b.copy() for b in biases]

    if rng is None:
        rng = np.random.default_rng(42)

    for epoch in range(epochs):

        for i in rng.permutation(len(x_train)):
            #Get the input pixels and the correct digit for the current training example
            pixels = x_train[i]
            correct_digit = y_train[i]
        
            #Calculate the scores for each layer
            scores, activations = forward(pixels, weights, biases)
        
            #Guess the digit by finding the index of the highest score in the output layer
            predicted_digit = np.argmax(scores[-1])
        
            #Store the output probabilities for each digit to train the model
            probabilities = softmax(scores[-1])

            #Calculate the gradient of the loss with respect to the output probabilities
            grad_scores = probabilities.copy()
            grad_scores[correct_digit] -= 1

            #Calculate the adjustment signals for the weights and biases of each layer using backpropagation
            grad_weights = [None] * len(weights)
            grad_biases = [None] * len(biases)
        
            for layer in reversed(range(len(weights))):
                grad_weights[layer] = np.outer(
                    activations[layer], grad_scores
                )
                grad_biases[layer] = grad_scores.copy()

                if layer > 0:
                    grad_activation = weights[layer] @ grad_scores

                    # Account for the previous hidden layer's ReLU.
                    grad_scores = (
                        grad_activation * (scores[layer - 1] > 0)
                    )

            #Update the weights and biases using the calculated gradients and the learning rate
            for layer in range(len(weights)):
                weights[layer] -= learning_rate * grad_weights[layer]
                biases[layer] -= learning_rate * grad_biases[layer]
        
        if printAccuracy:
            accuracy = test(training_data, weights, biases)
            print(
                f"Epoch {epoch + 1}: "
                f"training accuracy = {accuracy:.2%}"
            )

    return weights, biases

#Define the testing function to evaluate the accuracy of a neural network
def test(test_data, weights, biases):

    x_test, y_test = test_data
    correct_predictions = 0

    for pixels, correct_digit in zip(x_test, y_test):
        #Only care about the scores from each layer, not the activations, for testing.
        scores, _ = forward(pixels, weights, biases)
        predicted_digit = np.argmax(scores[-1])

        if predicted_digit == correct_digit:
            correct_predictions += 1

    return correct_predictions / len(x_test)

#Function to create random weights for a neural network given the sizes of each layer
def create_weights(layer_sizes, seed=42):
    rng = np.random.default_rng(seed)

    return [
        rng.normal(
            loc=0.0,
            scale=np.sqrt(2.0 / inputs),
            size=(inputs, outputs),
        )
        for inputs, outputs in zip(layer_sizes[:-1], layer_sizes[1:])
    ]

#Function to create biases for a neural network given the sizes of each layer
def create_biases(layer_sizes):

    return [
        np.zeros(size)
        for size in layer_sizes[1:]
    ]

#function to find the best neural network architecture and learning rate based on validation accuracy
def find_best_network(
    architectures,
    learning_rates,
    training_data,
    validation_data,
    max_epochs=30,
    seed=42,
    patience=5,
):
    best_network = None
    best_accuracy = -1.0
    best_settings = None

    for architecture in architectures:
        for learning_rate in learning_rates:
            network = NeuralNetwork(architecture, seed=seed)
            training_rng = np.random.default_rng(seed)
            run_best_accuracy = -1.0
            epochs_without_improvement = 0

            for epoch in range(1, max_epochs + 1):
                network.train(
                    training_data,
                    learning_rate=learning_rate,
                    epochs=1,
                    rng=training_rng,
                )

                parameters = network.weights + network.biases
                if not all(np.all(np.isfinite(p)) for p in parameters):
                    print(
                        f"Stopped unstable run: {architecture}, "
                        f"learning rate {learning_rate}"
                    )
                    break

                accuracy = network.test(validation_data)

                print(
                    f"{architecture} | "
                    f"LR: {learning_rate} | "
                    f"Epoch: {epoch} | "
                    f"Validation accuracy: {accuracy:.2%}"
                )

                if accuracy > run_best_accuracy:
                    run_best_accuracy = accuracy
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_settings = {
                        "architecture": list(architecture),
                        "learning_rate": learning_rate,
                        "epochs": epoch,
                    }

                    # Preserve this version before training continues.
                    best_network = NeuralNetwork(architecture, seed=seed)
                    best_network.weights = [
                        w.copy() for w in network.weights
                    ]
                    best_network.biases = [
                        b.copy() for b in network.biases
                    ]

                    np.savez(
                        "best_network.npz",
                        architecture=np.array(best_settings["architecture"]),
                        learning_rate=best_settings["learning_rate"],
                        epochs=best_settings["epochs"],
                        validation_accuracy=best_accuracy,
                        **{
                            f"weights_{i}": w
                            for i, w in enumerate(best_network.weights)
                        },
                        **{
                            f"biases_{i}": b
                            for i, b in enumerate(best_network.biases)
                        },
                    )

                if epochs_without_improvement >= patience:
                    print(
                        f"Early stopping at epoch {epoch}: "
                        f"no improvement for {patience} epochs."
                    )
                    break

    if best_network is None:
        raise ValueError("No successful network was evaluated.")

    return best_network, best_settings, best_accuracy

#split the training data into a training set and a validation set
def split_data(x_train, y_train, validation_proportion=0.2, seed=42):
    if not 0 < validation_proportion < 1:
        raise ValueError("validation_proportion must be between 0 and 1.")

    if len(x_train) != len(y_train):
        raise ValueError("Images and labels must have matching lengths.")

    validation_size = int(len(x_train) * validation_proportion)

    if validation_size < 1 or validation_size >= len(x_train):
        raise ValueError("The proportion must leave both sets nonempty.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x_train))

    validation_indices = indices[:validation_size]
    training_indices = indices[validation_size:]

    training_data = (
        x_train[training_indices],
        y_train[training_indices],
    )

    validation_data = (
        x_train[validation_indices],
        y_train[validation_indices],
    )

    return training_data, validation_data

#Function to display failed test cases for a neural network
def show_failed_tests(network, test_data, count=6):
    images, labels = test_data

    if count < 1:
        raise ValueError("count must be at least 1.")

    mistakes = []

    for pixels, correct_digit in zip(images, labels):
        scores, _ = forward(pixels, network.weights, network.biases)
        predicted_digit = int(np.argmax(scores[-1]))

        if predicted_digit != correct_digit:
            mistakes.append((pixels, correct_digit, predicted_digit))

        if len(mistakes) >= count:
            break

    if not mistakes:
        print("No failed test cases found.")
        return

    columns = min(3, len(mistakes))
    rows = (len(mistakes) + columns - 1) // columns

    figure, axes = plt.subplots(
        rows, columns,
        figsize=(3 * columns, 3 * rows),
        squeeze=False,
    )

    for axis in axes.flat:
        axis.axis("off")

    for axis, (pixels, actual, predicted) in zip(axes.flat, mistakes):
        axis.imshow(
            pixels.reshape(28, 28),
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axis.set_title(f"Actual: {actual}\nPredicted: {predicted}")

    figure.suptitle("Failed MNIST test cases")
    figure.tight_layout()
    plt.show()

class NeuralNetwork:
    def __init__(self, layer_sizes, seed=42):
        self.weights = create_weights(layer_sizes, seed)
        self.biases = create_biases(layer_sizes)

    def train(self, training_data, learning_rate, epochs, printAccuracy=False, rng=None):
        self.weights, self.biases = train(
            training_data,
            self.weights,
            self.biases,
            learning_rate,
            epochs,
            printAccuracy=printAccuracy,
            rng=rng,
        )

    def test(self, test_data):
        return test(test_data, self.weights, self.biases)

    def show_mistakes(self, test_data, count=6):
        images, labels = test_data
        mistakes = []

        if count < 1:
            raise ValueError("count must be at least 1")

        for pixels, correct_digit in zip(images, labels):
            scores, _ = forward(pixels, self.weights, self.biases)
            predicted_digit = int(np.argmax(scores[-1]))

            if predicted_digit != correct_digit:
                mistakes.append((pixels, correct_digit, predicted_digit))

            if len(mistakes) == count:
                break

        if not mistakes:
            print("No incorrectly predicted images found.")
            return

        columns = min(3, len(mistakes))
        rows = (len(mistakes) + columns - 1) // columns

        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(3 * columns, 3 * rows),
            squeeze=False,
        )

        for axis in axes.flat:
            axis.axis("off")

        for axis, (pixels, correct, predicted) in zip(axes.flat, mistakes):
            axis.imshow(
                pixels.reshape(28, 28),
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axis.set_title(f"Actual: {correct} | Predicted: {predicted}")

        figure.suptitle("Incorrect predictions")
        figure.tight_layout()
        plt.show()

training_data, validation_data = split_data(x_train, y_train, validation_proportion=0.2, seed=42)

"""Find the best neural network architecture and learning rate based on validation accuracy."""
"""# All architectures retain four layers: input, two hidden, output.
hidden_sizes = [16, 32, 64, 128, 256]

architectures = [
    [784, first_hidden, second_hidden, 10]
    for first_hidden in hidden_sizes
    for second_hidden in hidden_sizes
]

learning_rates = [
    0.0003,
    0.0005,
    0.001,
    0.002,
    0.003,
    0.005,
    0.01,
    0.02,
    0.03,
]

architectures = [[784, 64, 32, 10]]

architectures = [
    [784, 128, 64, 10]
]

learning_rates = [0.001, 0.003, 0.01]

best_network, best_settings, validation_accuracy = find_best_network(
    architectures=architectures,
    learning_rates=learning_rates,
    training_data=training_data,
    validation_data=validation_data,
    max_epochs=30,
    patience=7,
)

print("Best settings:", best_settings)
print(f"Best validation accuracy: {validation_accuracy:.2%}")

test_accuracy = best_network.test((x_test, y_test))
print(f"Final test accuracy: {test_accuracy:.2%}")"""

myNeuralNetwork = NeuralNetwork([784, 128, 64, 10],)

myNeuralNetwork.train(
    training_data,
    learning_rate=0.003,
    epochs=19,
    printAccuracy=True,
)

accuracy = myNeuralNetwork.test((x_test, y_test))
print(f"Test accuracy: {accuracy:.2%}")

show_failed_tests(myNeuralNetwork, (x_test, y_test), count=9)


