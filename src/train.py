from data_loader import load_data
from model import (
    gradient_descent,
    forward_prop,
    get_predictions,
    get_accuracy
)

DATASET_PATH = "data/mnist_train.csv"

def main():
    print("Loading data...")
    X_train, Y_train, X_dev, Y_dev = load_data(DATASET_PATH)

    print("Training model...")
    W1, b1, W2, b2 = gradient_descent(
        X_train,
        Y_train,
        alpha=0.1,
        iterations=500
    )

    print("Evaluating on dev set...")
    _, _, _, A2_dev = forward_prop(W1, b1, W2, b2, X_dev)
    dev_predictions = get_predictions(A2_dev)
    dev_accuracy = get_accuracy(dev_predictions, Y_dev)

    print(f"Final Dev Accuracy: {dev_accuracy:.4f}")

if __name__ == "__main__":
    main()
