import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def mean_absolate_error(prediction_values, actual_values):
    accumulated_error = 0.0
    for prediction_value, actual_value in zip(prediction_values, actual_values):
        accumulated_error += np.abs(prediction_values - actual_values)
    
    # calculated mean
    mae_error = accumulated_error / len(prediction_values)
    return mae_error

def mean_squared_error(prediction_values, actual_values):
    accumulated_error = 0.0
    for prediction, actual in zip(prediction_values, actual_values):
        accumulated_error += (prediction - actual)**2
    
    mae_error = (1.0 / (2 * len(prediction_values))) * accumulated_error

    return mae_error

def gradient_decent(x, y):
    '''
    : Calculate global minimum
    : https://www.quora.com/Does-Gradient-Descent-Algo-always-converge-to-the-global-minimum
    '''
    m_curr, b_curr = 6, 30
    total_iteration = 10000
    learning_rate = 0.0003

    for _ in range(total_iteration):
        y_predicts = m_curr * x + b_curr
        cost = (1.0 / (2 * len(x))) * sum((y_predicts - y)**2) # calculate mean_squared_error
        # calculate prtial derivatives using cost function to change m & b value
        # gives me slope
        m_derivatives = -(2/len(x)) * sum(x * (y - y_predicts)) 
        b_derivatives = -(2/len(x)) * sum(y - y_predicts)
        # for direction and step
        m_curr = m_curr - learning_rate * m_derivatives
        b_curr = b_curr - learning_rate * b_derivatives

        #check if fit is "good enough"
        if cost < 0:
            break

    print("gradient_decent:: m = {}, b = {}, cost = {}".format(m_curr, b_curr, cost))
    return m_curr, b_curr

def draw_regression_line(x, y):
    '''
    :calculate m(slope) , b(y intersect)
    '''
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    sdx, sdx_sdy = 0.0, 0.0
    for i,j in zip(x, y):
        sdx += (i - x_mean) ** 2
        sdx_sdy += (i - x_mean) * (j - y_mean)

    m = sdx_sdy / sdx
    b = y_mean - (m * x_mean)

    return m, b

def predict(x, parameters):
    '''
    :x is feature array
    :parameters holds `weight` and y intersect value `b`
    :calculate y = sum(w.x) + b
    '''
    # prediction = 0
    # for weight, feature in zip(parameters["w"], x):
    #     prediction += weight * feature
    
    # # adding bias
    # prediction += parameters["b"]
    prediction = (parameters["w"] * x) + parameters["b"]
    return prediction

def linear_regression():
    df_data = pd.read_csv("./cracow_apartments.csv", sep=",")
    features = ["size"]
    target = ["price"]
    
    # separate feature vectors and target value
    x, y = df_data[features].to_numpy(), df_data[target].to_numpy()

    # ploting 
    fig, ax1 = plt.subplots()
    ax1.set_title("Cracow Apartments Price by Size")
    ax1.set_xlabel(features[0])
    ax1.set_xlabel(target[0])
    ax1.plot(x, y, "go")

    m, b = draw_regression_line(x, y)

    line_pram = {"w": m, "b": b}
    pred_values = [predict(i, line_pram) for i in x]

    # model error
    mse_error = mean_squared_error(pred_values, y)
    print("mean_squared_error:: m = {}, b = {}, cost = {}".format(m, b , mse_error))

    mm, bb = gradient_decent(x,y)
    line_pram = {"w": mm, "b": bb}
    pred_values = np.array([predict(i, line_pram) for i in x])

    # predicted value ploting
    ax2 = ax1.twinx()
    ax2.plot(x, pred_values, color = "red", linestyle = "solid")
    fig.tight_layout()
    plt.show()

    t = 0
    for i, j in zip(y, pred_values):
        print("House size = {}, actual price = {}, predicted price = {}".format(x[t], i, j))
        t += 1
    

if __name__ == "__main__":
    linear_regression()
        