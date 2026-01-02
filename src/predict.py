import matplotlib.pyplot as plt
from data_loader import load_data
from model import gradient_descent, forward_prop, get_predictions

DATASET_PATH = "data/mnist_train.csv"

def main():
    X_train, Y_train, _, _ = load_data(DATASET_PATH)

    W1, b1, W2, b2 = gradient_descent(
        X_train,
        Y_train,
        alpha=0.1,
        iterations=300
    )

    index = 0
    image = X_train[:, index, None]

    _, _, _, A2 = forward_prop(W1, b1, W2, b2, image)
    prediction = get_predictions(A2)[0]

    plt.imshow(image.reshape(28, 28), cmap="gray")
    plt.title(f"Prediction: {prediction}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
