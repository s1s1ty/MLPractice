import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def init(n):
    return {"w": np.zeros(n), "b": 0.0}

def predict(x, parameters):
    '''
    :x is feature array
    :parameters holds `weight` and y intersect value
    :calculate y = sum(w.x) + b
    '''
    prediction = 0
    print(parameters["w"])
    for weight, feature in zip(parameters["w"], x):
        prediction += weight * feature
    
    # adding bias
    prediction += parameters["b"]
    return prediction

def mean_absolate_error(prediction_values, target_values):
    accumulated_error = 0.0
    for prediction_value, target_value in zip(prediction_values, target_values):
        accumulated_error += np.abs(prediction_values - target_values)
    
    # calculated mean
    mae_error = accumulated_error / len(prediction_values)
    return mae_error

def mean_squared_error(prediction_values, target_values):
    accumulated_error = 0.0
    
    for prediction, target in zip(prediction_values, target_values):
        accumulated_error += (prediction - target)**2
    
    mae_error = (1.0 / (2 * len(prediction_values))) * accumulated_error

    return mae_error
    

def main():
    df_data = pd.read_csv("./cracow_apartments.csv", sep=",")

    features = ["size"]
    target = ["price"]
    
    # separate feature vectors and target value
    x, y = df_data[features].to_numpy(), df_data[target].to_numpy()
    
    or_pram = {"w": np.array([3.0]), "b": 200}
    lime_pram = {"w": np.array([12.0]), "b": -160}
    
    or_pred = [predict(x, or_pram) for i in x]
    lime_pred = [predict(x, lime_pram) for i in x]
            
    # model error
    mse_or_error = mean_squared_error(or_pred, y)
    mse_lime_error = mean_squared_error(lime_pred, y)
    
    print(mse_or_error, mse_lime_error)
    

if __name__ == "__main__":
    main()
        