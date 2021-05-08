import pandas as pd
import numpy as np

total_iteration = 10000
learning_rate = 0.00001


def fit_line(x, y, total_feature):
    m = np.ones(total_feature)
    x_transposed = x.transpose()
    y_transposed = y.transpose()

    for _ in range(total_iteration):
        y_predicts = np.dot(x, m)
        y_predicts_diff = y_predicts - y_transposed[0]
        cost = (1.0 / (2 * total_feature)) * np.sum(y_predicts_diff ** 2)
        # avg gradient per example
        gradient = np.dot(x_transposed, y_predicts_diff) / total_feature
        m = m - learning_rate * gradient

    return m

def predict(x, m):
    return np.dot(x, m)

def main():
    data = pd.read_csv("./cracow_apartments.csv", sep=",")
    features = ["rooms", "size", "distance_to_city_center"]
    target = ["price"]

    # Seperate feature vector and target values
    x, y = data[features].to_numpy(), data[target].to_numpy()

    # x_vector = np.column_stack(x)
    m = fit_line(x, y, len(features))
    pridected_y = predict(x, m)
    print(x, m, pridected_y)

if __name__ == "__main__":
    main()