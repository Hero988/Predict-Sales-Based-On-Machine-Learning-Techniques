# Import necessary libraries
import numpy as np  # For numerical operations
import pandas as pd  # For handling datasets
import torch  # PyTorch library for tensor operations and neural networks
import torch.nn as nn  # Module of PyTorch containing neural network layers
from sklearn.preprocessing import StandardScaler  # For feature scaling
from torch.utils.data import DataLoader, TensorDataset  # For creating data loaders and tensor datasets
import matplotlib.pyplot as plt  # Import matplotlib.pyplot for plotting graphs

from werkzeug.utils import secure_filename

from flask_mail import Mail, Message

import mysql.connector

import json

import stripe

from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.nn.functional as F


from flask import Flask, request, jsonify, send_file, render_template, Response, redirect, url_for, session

from werkzeug.security import generate_password_hash, check_password_hash

from sklearn.linear_model import LinearRegression

import logging

import traceback

import xgboost as xgb

from datetime import datetime, timedelta

import io

import string
import random

import process_data_helper

import pickle

import os

from prophet import Prophet

import glob

from flask_cors import CORS

from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Multilayer Perceptron
class SalesPredictionMLP(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super(SalesPredictionMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.output_layer(x)
        return x

def get_user_date_input(prompt):
    # Sets the date format
    date_format = '%Y-%m-%d'
    date_str = input(prompt)
    # Try to convert the user input into a datetime object to validate it
    while True:
        try:
            # If successful, return the string as it is valid
            pd.to_datetime(date_str, format=date_format)
            return date_str
        except ValueError:
            # If there's a ValueError, it means the format is incorrect. Prompt the user again.
            print("The date format is incorrect. Please enter the date in 'YYYY-MM-DD' format.")
            date_str = input(prompt)

def evaluate_model(model, loader, test_dataframe, choice, model_saved, action, db_name):
    if model_saved is not None and action != 'train':
        model.load_state_dict(torch.load(model_saved))
    elif model_saved is None and action != 'train':
        conn = None
        try:
            conn = process_data_helper.connect_to_db(db_name)
            model_state = process_data_helper.retrieve_model_from_db(conn, model_name="MyCustomModelName")
            if model_state:
                model.load_state_dict(model_state)
            else:
                print("Model state not found in the database.")
        finally:
            if conn:
                conn.close()

    model.eval()  # Set the model to evaluation mode

    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs).squeeze().cpu().numpy()
            predictions.extend(outputs)
            actuals.extend(targets.numpy())

    test_dataframe['Predicted'] = predictions

    # Creating 'Previous Predicted' column
    test_dataframe['Previous Predicted'] = test_dataframe['Predicted'].shift(1).fillna(0)

    # Calculate predicted movement based on the new 'Predicted' values
    test_dataframe['Predicted Movement'] = np.where(test_dataframe['Predicted'] > test_dataframe['Previous Predicted'], 1,
                                                    np.where(test_dataframe['Predicted'] < test_dataframe['Previous Predicted'], -1, 0))

    # Calculate evaluation metrics
    mse_error = mean_squared_error(test_dataframe['sales_data_current'], test_dataframe['Predicted'])
    movement_accuracy = accuracy_score(test_dataframe['Actual Movement'], test_dataframe['Predicted Movement'])

    # Save predictions to database
    conn = process_data_helper.connect_to_db(db_name)
    result_id = "result_id123"
    experiment_id = "experiment_id123"
    process_data_helper.save_predictions_to_db(conn, result_id, experiment_id, test_dataframe)
    conn.close()

    movement_accuracy_percent = movement_accuracy * 100

    print(f"Mean Squared Error: {mse_error:.4f}")
    print(f"Movement Accuracy: {movement_accuracy_percent:.2f}% - Indicates how well the model predicted sales direction changes.")

    return mse_error, movement_accuracy_percent, test_dataframe

def get_start_end_date_of_csv(filename):
    # Read the CSV file into a pandas DataFrame.
    data = pd.read_csv(filename)

    # Set the 'datetime' column as the index of the DataFrame.
    data.set_index('date', inplace=True)

    # Convert the index into a datetime format for time-series manipulation.
    data.index = pd.to_datetime(data.index)

    # Find the minimum and maximum dates in the index
    start_date = data.index.min().strftime('%Y-%m-%d')
    end_date = data.index.max().strftime('%Y-%m-%d')

    return start_date, end_date

def load_models(model_paths, model):
    models = []
    for path in model_paths:
        model.load_state_dict(torch.load(path))
        models.append(model)
    return models

def split_data_for_prophet_evaluation(retrieved_dataset, split_ratio=0.8):

    # Ensure 'date' column is in datetime format
    retrieved_dataset['date'] = pd.to_datetime(retrieved_dataset['date'])

    # Rename columns to fit Prophet's expected column names
    prophet_df = retrieved_dataset.rename(columns={'date': 'ds', 'sales_data_current': 'y'})

    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

    # Remove timezone information
    prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)

    # Replace any NaN values with 0
    prophet_df.fillna(0, inplace=True)

    # Ensure the dataset is sorted by date
    prophet_df.sort_values('ds', inplace=True)

    # Calculate the split index
    split_index = int(len(prophet_df) * split_ratio)

    # Split the dataset into training and testing sets
    train_set = prophet_df.iloc[:split_index]
    test_set = prophet_df.iloc[split_index:]

    return train_set, test_set

def split_data_for_xgboost_evaluation(retrieved_dataset, split_ratio=0.8):
    # Ensure 'date' column is in datetime format
    retrieved_dataset['date'] = pd.to_datetime(retrieved_dataset['date'])

    xgboost_dataset = retrieved_dataset.copy()

    # Ensure the dataset is sorted by date
    xgboost_dataset.sort_values('date', inplace=True)

    # Calculate the split index
    split_index = int(len(xgboost_dataset) * split_ratio)

    # Split the dataset into training and testing sets
    train_set = xgboost_dataset.iloc[:split_index]
    test_set = xgboost_dataset.iloc[split_index:]

    return train_set, test_set

def predict_and_evaluate_model(model_prophet, test_df):
    df_for_prediction = test_df.drop(columns=['y'])
    forecast = model_prophet.predict(df_for_prediction)

    # Evaluation metrics
    mae = mean_absolute_error(test_df['y'], forecast['yhat'])
    rmse = np.sqrt(mean_squared_error(test_df['y'], forecast['yhat']))  # Use np.sqrt to get RMSE
    mape = mean_absolute_percentage_error(test_df['y'], forecast['yhat'])

    # Calculate Actual Movement
    test_df['Actual Movement'] = np.sign(test_df['y'].diff().fillna(0))

    # Calculate Predicted Movement
    forecast['Predicted Movement'] = np.sign(forecast['yhat'].diff().fillna(0))

    # Ensure the first value of 'Predicted Movement' isn't NaN due to the diff operation
    forecast['Predicted Movement'] = forecast['Predicted Movement'].fillna(0)

    # Map predicted movements to the corresponding dates in test_df
    test_df['Predicted Movement'] = test_df['ds'].map(forecast.set_index('ds')['Predicted Movement']).fillna(0)

    # Calculate movement prediction accuracy
    movement_accuracy = accuracy_score(test_df['Actual Movement'], test_df['Predicted Movement']) * 100  # Convert to percentage

    rmse_fixed = f"{rmse:.2f}"

    print(f"MAE: {mae}, RMSE: {rmse}, MAPE: {mape}, Movement Accuracy: {movement_accuracy:.2f}%")

    return mae, rmse_fixed, mape, movement_accuracy, forecast, test_df

def evaluate_predictions_xgboost(df):

    # Replace infinite values with NaN as the NAN are just sales_data_current with no predidctions
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # We remove those
    df.dropna(subset=['sales_data_current', 'prediction'], inplace=True)

    rmse = np.sqrt(mean_squared_error(df['sales_data_current'], df['prediction']))  # Use np.sqrt to get RMSE

    # Calculate Actual Movement
    df['Actual Movement'] = np.sign(df['sales_data_current'].diff().fillna(0))

    # Calculate Predicted Movement
    df['Predicted Movement'] = np.sign(df['prediction'].diff().fillna(0))

    # Ensure the first value of 'Predicted Movement' isn't NaN due to the diff operation
    df['Predicted Movement'] = df['Predicted Movement'].fillna(0)

    # Map predicted movements to the corresponding dates in test_df
    df['Predicted Movement'] = df['date'].map(df.set_index('date')['Predicted Movement']).fillna(0)

    # Calculate movement prediction accuracy
    movement_accuracy = accuracy_score(df['Actual Movement'], df['Predicted Movement']) * 100  # Convert to percentage

    rmse_fixed = f"{rmse:.2f}"

    print(f"RMSE: {rmse}, Movement Accuracy: {movement_accuracy:.2f}%")

    return rmse_fixed, df, movement_accuracy

def save_predictions_plot(df, db_name, action):
    if action == 'evaluate_train':
        plot_name = 'LSTM_plot'
    elif action == 'xgboost':
        plot_name = 'xgboost_plot'
    elif action == 'xgboost-prediction':
        plot_name = 'xgboost_plot-prediction'
    elif action == 'predict-next':
        plot_name = 'predict-next_plot'
    elif action == 'prophet-predict':
        plot_name = 'prophet-predict_plot'
    elif action == 'ensemble':
        plot_name = 'ensemble_evaluate_plot'
    elif action == 'ensemble_prediction':
        plot_name = 'ensemble_prediction_plot'

    # Buffer to save image
    buf = io.BytesIO()

    if action == 'xgboost' or action == 'ensemble':
        plt.figure(figsize=(14, 8))

        # Plotting Sales Forecast vs Actuals
        actual_dates = df['date']
        plt.subplot(2, 1, 1)  # This means 2 rows, 1 column, position 1
        plt.plot(actual_dates, df['sales_data_current'], label='Actual Sales', marker='o', color='blue')
        plt.plot(actual_dates, df.loc[df['date'].isin(actual_dates), 'prediction'], label='Forecasted Sales', marker='x', color='red')
        plt.title('Sales Forecast vs Actuals on 20% off provided data')
        plt.ylabel('Sales')
        plt.legend()

        # Plotting Actual vs Predicted Movements
        plt.subplot(2, 1, 2)  # This means 2 rows, 1 column, position 2
        plt.scatter(actual_dates, df['Actual Movement'], color='blue', label='Actual Movement', alpha=0.5, edgecolor='black')
        plt.scatter(actual_dates, df['Predicted Movement'], color='red', label='Predicted Movement', marker='x', alpha=0.5)
        # Highlighting differences
        differences = df['Actual Movement'] != df['Predicted Movement']
        plt.scatter(actual_dates[differences], df['Predicted Movement'][differences], color='yellow', label='Differences', marker='o', facecolors='none', s=100)
        plt.title('Actual vs Predicted Movement on 20% off provided data')
        plt.xlabel('Date')
        plt.ylabel('Movement')
        plt.xticks(rotation=45)
        plt.legend()

        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()

        buf.seek(0)

        conn = process_data_helper.connect_to_db(db_name)
        process_data_helper.save_plot_to_db(db_name, plot_name, buf.getvalue())
        conn.close()
        print(f"Plot {plot_name} saved to database successfully.")
    elif action == 'evaluate_train':
        plt.figure(figsize=(14, 8))

        # Plotting Sales Forecast vs Actuals
        actual_dates = df['date']
        plt.subplot(2, 1, 1)  # This means 2 rows, 1 column, position 1
        plt.plot(actual_dates, df['sales_data_current'], label='Actual Sales', marker='o', color='blue')
        plt.plot(actual_dates, df.loc[df['date'].isin(actual_dates), 'Predicted'], label='Forecasted Sales', marker='x', color='red')
        plt.title('Sales Forecast vs Actuals on 20% off provided data')
        plt.ylabel('Sales')
        plt.legend()

        # Plotting Actual vs Predicted Movements
        plt.subplot(2, 1, 2)  # This means 2 rows, 1 column, position 2
        plt.scatter(actual_dates, df['Actual Movement'], color='blue', label='Actual Movement', alpha=0.5, edgecolor='black')
        plt.scatter(actual_dates, df['Predicted Movement'], color='red', label='Predicted Movement', marker='x', alpha=0.5)
        # Highlighting differences
        differences = df['Actual Movement'] != df['Predicted Movement']
        plt.scatter(actual_dates[differences], df['Predicted Movement'][differences], color='yellow', label='Differences', marker='o', facecolors='none', s=100)
        plt.title('Actual vs Predicted Movement on 20% off provided data')
        plt.xlabel('Date')
        plt.ylabel('Movement')
        plt.xticks(rotation=45)
        plt.legend()

        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()

        buf.seek(0)

        conn = process_data_helper.connect_to_db(db_name)
        process_data_helper.save_plot_to_db(db_name, plot_name, buf.getvalue())
        conn.close()
    elif action == 'xgboost-prediction' or action == 'predict-next' or action == 'ensemble_prediction' or action == 'prophet-predict':
        # Using a named color
        df['pred'].plot(figsize=(10, 5), color='magenta', linestyle='-', marker='o', markersize=1, linewidth=1, title='Future Predictions')
        plt.legend()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()

        buf.seek(0)

        conn = process_data_helper.connect_to_db(db_name)
        process_data_helper.save_plot_to_db(db_name, plot_name, buf.getvalue())
        conn.close()

def save_forecast_plot(forecast, test_df, db_name):
    plot_name = 'prophet_plot'
    # Buffer to save image
    buf = io.BytesIO()

    plt.figure(figsize=(14, 8))

    # Plotting Sales Forecast vs Actuals
    actual_dates = test_df['ds']
    plt.subplot(2, 1, 1)  # This means 2 rows, 1 column, position 1
    plt.plot(actual_dates, test_df['y'], label='Actual Sales', marker='o', color='blue')
    plt.plot(actual_dates, forecast.loc[forecast['ds'].isin(actual_dates), 'yhat'], label='Forecasted Sales', marker='x', color='red')
    plt.fill_between(actual_dates, forecast.loc[forecast['ds'].isin(actual_dates), 'yhat_lower'], forecast.loc[forecast['ds'].isin(actual_dates), 'yhat_upper'], color='gray', alpha=0.2)
    plt.title('Sales Forecast vs Actuals on 20% off provided data')
    plt.ylabel('Sales')
    plt.legend()

    # Plotting Actual vs Predicted Movements
    plt.subplot(2, 1, 2)  # This means 2 rows, 1 column, position 2
    plt.scatter(actual_dates, test_df['Actual Movement'], color='blue', label='Actual Movement', alpha=0.5, edgecolor='black')
    plt.scatter(actual_dates, test_df['Predicted Movement'], color='red', label='Predicted Movement', marker='x', alpha=0.5)
    # Highlighting differences
    differences = test_df['Actual Movement'] != test_df['Predicted Movement']
    plt.scatter(actual_dates[differences], test_df['Predicted Movement'][differences], color='yellow', label='Differences', marker='o', facecolors='none', s=100)
    plt.title('Actual vs Predicted Movement on 20% off provided data')
    plt.xlabel('Date')
    plt.ylabel('Movement')
    plt.xticks(rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()

    buf.seek(0)

    conn = process_data_helper.connect_to_db(db_name)
    process_data_helper.save_plot_to_db(db_name, plot_name, buf.getvalue())
    conn.close()
    print(f"Plot {plot_name} saved to database successfully.")

def find_model_paths(base_folder):
    model_paths = []

    # Loop through each item in the base folder
    for root, dirs, files in os.walk(base_folder):
        for dir in dirs:
            # Construct the path to the subdirectory
            subdirectory_path = os.path.join(root, dir)
            # Use glob to find .pth files in the subdirectory
            for model_path in glob.glob(os.path.join(subdirectory_path, "*.pth")):
                model_paths.append(model_path)

    return model_paths

def feature_importance_xgboost(reg, db_name):
    plot_name = 'xgboost_feature_importance'  # Corrected the plot name to match the context
    # Buffer to save image
    buf = io.BytesIO()

    # Explicitly set a larger figure size
    plt.figure(figsize=(10, len(reg.feature_names_in_) * 0.5))  # Adjust the height based on the number of features

    # Create DataFrame for feature importances
    fi = pd.DataFrame(data=reg.feature_importances_,
                      index=reg.feature_names_in_,
                      columns=['importance'])

    # Sort and plot feature importance
    fi.sort_values('importance', ascending=True).plot(kind='barh', title='Feature Importance', figsize=(10, 8))

    plt.tight_layout()  # Adjust subplots to fit into the figure area.
    plt.savefig(buf, format='png')
    plt.close()

    buf.seek(0)

    # Save the plot to the database
    conn = process_data_helper.connect_to_db(db_name)
    process_data_helper.save_plot_to_db(db_name, plot_name, buf.getvalue())
    conn.close()
    print(f"Plot {plot_name} saved to database successfully.")

def save_predictions_to_database(data, db_name):
    # Convert array to pandas DataFrame
    df = pd.DataFrame(data)

    conn = process_data_helper.connect_to_db(db_name)

    process_data_helper.save_dataset_to_db(conn, 'saved_predictions', 'User Provided Dataset', df)

def add_features(df):

    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear

    return df

def prepare_data_for_MLP_time_series(retrieved_dataset, date_column, target_name, validation_size, db_name):

    if date_column not in retrieved_dataset.columns:
        raise ValueError(f"{date_column} not found in DataFrame columns.")

    retrieved_dataset[date_column] = pd.to_datetime(retrieved_dataset[date_column])
    retrieved_dataset.sort_values(by=date_column, inplace=True)

    # Before creating X_train and X_validation, ensure retrieved_dataset is ready for final processing

    retrieved_dataset.drop(columns=[date_column, 'Actual Movement', 'sales_data_percentage_change'], inplace=True)  # Drop the date_column if it's not converted to a numeric feature

    dropped_columns = []

    # Identify non-numeric columns to drop, excluding the target column
    for col in retrieved_dataset.columns:
        if retrieved_dataset[col].dtype == 'object' and col != target_name:
            dropped_columns.append(col)

    # Drop the identified columns
    retrieved_dataset.drop(columns=dropped_columns, inplace=True)

    split_index = int(len(retrieved_dataset) * (1 - validation_size))
    train_dataset = retrieved_dataset.iloc[:split_index]
    validation_dataset = retrieved_dataset.iloc[split_index:]

    # Directly use DataFrames for target separation and feature scaling
    X_train_df = train_dataset.drop(columns=[target_name])
    y_train = train_dataset[target_name].values
    X_validation_df = validation_dataset.drop(columns=[target_name])
    y_validation = validation_dataset[target_name].values

    scaler = StandardScaler()
    # Fit and transform training data, transform validation data
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_validation_scaled = scaler.transform(X_validation_df)

    # Save the scaler - adjust the function as needed based on how you've implemented saving scalers
    scaler_name = 'scaler'
    experiment_id = generate_experiment_id(10)  # Generate a 10-character string
    conn = process_data_helper.connect_to_db(db_name)
    process_data_helper.save_scaler_to_db(conn, scaler_name, scaler)
    conn.close()

    # Your return statement remains the same, just converting the scaled arrays to tensors
    return {
        "X_train": torch.tensor(X_train_scaled).float(),
        "X_validation": torch.tensor(X_validation_scaled).float(),
        "y_train": torch.tensor(y_train).float(),
        "y_validation": torch.tensor(y_validation).float(),
        "dropped_columns": dropped_columns  # Return the list of dropped columns
    }

def calculate_movement(data_series):
    # Using numpy's diff function to calculate the difference between consecutive elements
    return np.diff(data_series, prepend=data_series[0]) > 0

def generate_experiment_id(length=8):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for i in range(length))

def train_until_target_movement_accuracy(model, criterion, optimizer, X_train, y_train, X_val, y_val, target_accuracy, patience, db_name):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_movement_accuracy = 0.0
    epochs_no_improve = 0
    epoch = 0

    print("Starting training...")

    while best_movement_accuracy < target_accuracy:
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions.squeeze(), y_train)  # Ensure y_train is correctly shaped
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            predictions_val = model(X_val).squeeze()  # Remove extra dimensions from predictions
            loss_val = criterion(predictions_val, y_val)
            val_losses.append(loss_val.item())


            # Convert to numpy arrays for movement calculation
            actuals_array = y_val.numpy()
            predictions_array = predictions_val.numpy()

            # Calculate movements
            actual_movements = calculate_movement(actuals_array)
            predicted_movements = calculate_movement(predictions_array)

            # Calculate movement accuracy
            movement_accuracy = accuracy_score(actual_movements, predicted_movements)


            # Check if movement accuracy is at a new best and update if needed
            if movement_accuracy > best_movement_accuracy:
                best_movement_accuracy = movement_accuracy
                # Usage example, integrated with checking, deletion, and saving of a new model
                conn = process_data_helper.connect_to_db(db_name)  # Make sure you define or replace `connect_to_db` with your actual database connection function
                model_name = 'MP_model'
                experiment_id = "Experiment123"

                # Check if the model already exists in the database
                existing_model = process_data_helper.retrieve_model_from_db(conn, model_name=model_name)
                if existing_model is not None:
                    # Delete the existing model from the database
                    process_data_helper.delete_model_from_db(conn, model_name)

                # Save the new model to the database
                process_data_helper.save_model_to_db(conn, model_name, model, experiment_id)

                conn.close()  # Always remember to close the database connection when done

                best_val_loss = loss_val
                epochs_no_improve = 0

                if best_movement_accuracy >= target_accuracy:
                    break  # Exit the loop if target accuracy is reached



            """
          # Check for early stopping condition
            if loss_val < best_val_loss:
                # Usage example, integrated with checking, deletion, and saving of a new model
                conn = process_data_helper.connect_to_db(db_name)  # Make sure you define or replace `connect_to_db` with your actual database connection function
                model_name = 'MP_model'
                experiment_id = "Experiment123"

                # Check if the model already exists in the database
                existing_model = process_data_helper.retrieve_model_from_db(conn, model_name=model_name)
                if existing_model is not None:
                    # Delete the existing model from the database
                    process_data_helper.delete_model_from_db(conn, model_name)

                # Save the new model to the database
                process_data_helper.save_model_to_db(conn, model_name, model, experiment_id)

                conn.close()  # Always remember to close the database connection when done

                best_val_loss = loss_val
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

            """

        epoch += 1

    print("Training completed.")

def split_dataframe(dataframe, split_ratio=0.8):
    # Calculate the split index
    split_index = int(len(dataframe) * split_ratio)

    # Split the dataset
    training_set = dataframe.iloc[:split_index]
    testing_set = dataframe.iloc[split_index:]

    return training_set, testing_set

def prepare_data_for_evaluation(retrieved_dataset, date_column, target_name, scaler, dropped_columns):
    if date_column not in retrieved_dataset.columns:
        raise ValueError(f"{date_column} not found in DataFrame columns.")

    retrieved_dataset[date_column] = pd.to_datetime(retrieved_dataset[date_column])
    retrieved_dataset.sort_values(by=date_column, inplace=True)

    dropped_columns = []  # Keep track of dropped columns

    # Identify non-numeric columns to drop, excluding the target column
    for col in retrieved_dataset.columns:
        if retrieved_dataset[col].dtype == 'object' and col != target_name:
            dropped_columns.append(col)

    # Exclude specified columns and any non-numeric columns, plus the target column itself
    excluded_columns = [target_name, date_column, 'Actual Movement', 'sales_data_percentage_change'] + (dropped_columns or [])
    excluded_columns += list(retrieved_dataset.select_dtypes(exclude=[np.number]).columns)
    # Corrected call to copy()
    features = retrieved_dataset.drop(columns=excluded_columns).copy()

    # Now, scale features
    features_scaled = scaler.transform(features)

    # Extract the actual values for the target variable, correctly aligned with the features
    actuals = retrieved_dataset[target_name].values

    # Converting scaled data to tensors for PyTorch and returning actuals as a NumPy array
    return torch.tensor(features_scaled).float(), actuals

def plot_predictions_vs_actuals(actuals, predictions, db_name):
    title='Sales Prediction vs Actuals'
    plot_name = 'MLP_plot_predictions_vs_actuals'
    # Calculate movement
    actual_movements = calculate_movement(actuals)
    predicted_movements = calculate_movement(predictions)

    # Buffer to save image
    buf = io.BytesIO()

    # Calculate accuracy of movements
    movement_accuracy = accuracy_score(actual_movements, predicted_movements)
    movement_accuracy_percentage = "{:.2%}".format(movement_accuracy)

    print(f"Movement Accuracy: {movement_accuracy:.2%}")

    # Plot actual vs predicted sales
    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
    plt.plot(actuals, label='Actual Sales', color='blue', marker='o')
    plt.plot(predictions, label='Predicted Sales', color='red', linestyle='--', marker='x')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Sales')
    plt.legend()

    # Plot actual vs predicted movements
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
    plt.plot(actual_movements, label='Actual Movement', color='blue', linestyle='None', marker='o')
    plt.plot(predicted_movements, label='Predicted Movement', color='red', linestyle='None', marker='x')
    plt.title('Movement Prediction vs Actuals')
    plt.xlabel('Time')
    plt.ylabel('Movement')
    plt.legend()

    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()

    buf.seek(0)

    conn = process_data_helper.connect_to_db(db_name)
    process_data_helper.save_plot_to_db(db_name, plot_name, buf.getvalue())
    conn.close()
    print(f"Plot {plot_name} saved to database successfully.")

    return movement_accuracy_percentage

def run_multiple_experiments(choice, action, db_name, number_of_days):
    # Initialize return variables
    mae, rmse, mape = None, None, None
    target_name = 'sales_data_current'
    feature_names = ['dayofweek', 'quarter', 'month', 'year', 'dayofyear']
    model_saved = None

    if action == 'prophet':
        dataset_id = 'processed_data_for_prophet'
        conn = process_data_helper.connect_to_db(db_name)
        retrieved_dataset = process_data_helper.retrieve_dataset_from_db(conn, dataset_id)
        conn.close()
    elif action == 'xgboost':
        # Assuming 'processed_data' is the ID for the latest dataset
        dataset_id = "processed_data_for_xgboost"
        conn = process_data_helper.connect_to_db(db_name)
        # Retrieve the latest dataset for prediction
        retrieved_dataset = process_data_helper.retrieve_dataset_from_db(conn, dataset_id)
    elif action == 'train':
        conn = process_data_helper.connect_to_db(db_name)
        dataset_id = "processed_data"
        retrieved_dataset = process_data_helper.retrieve_dataset_from_db(conn, dataset_id)
        conn.close()

    if action == 'train':
        training_set, testing_set = split_dataframe(retrieved_dataset)

        date_column = 'date'
        target_name = 'sales_data_current'

        validation_size = 0.2

        data_preparation_result = prepare_data_for_MLP_time_series(training_set, date_column, target_name, validation_size, db_name)

        # Unpack necessary items from data preparation result
        X_train = data_preparation_result["X_train"]
        X_validation = data_preparation_result["X_validation"]
        y_train = data_preparation_result["y_train"]
        y_validation = data_preparation_result["y_validation"]
        dropped_columns = data_preparation_result["dropped_columns"]  # List of dropped columns

        # Step 4: Define and train the model
        input_size = X_train.shape[1]  # Feature size
        hidden_size = 64
        dropout_rate = 0.2

        model = SalesPredictionMLP(input_size, hidden_size, dropout_rate)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Within run_time_series_forecasting function
        train_until_target_movement_accuracy(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            X_train=X_train,
            y_train=y_train,
            X_val=X_validation,
            y_val=y_validation,
            target_accuracy=0.8,  # Assuming you want to use 0.8 as the target accuracy
            patience=20, # Increase as needed
            db_name=db_name
        )

        scaler_name = 'scaler'

        conn = process_data_helper.connect_to_db(db_name)

        scaler = process_data_helper.load_scaler_from_db(conn, scaler_name)
        # Corrected to use test_dataset for preparation
        X_test_tensor, actuals_array = prepare_data_for_evaluation(testing_set, date_column, target_name, scaler, dropped_columns)

        model.eval()
        with torch.no_grad():
            predictions = model(X_test_tensor)

        mse = mean_squared_error(actuals_array, predictions.numpy())

        # Assuming predictions and actuals have been calculated
        predictions_array = predictions.squeeze().numpy()

        movement_accuracy = plot_predictions_vs_actuals(actuals_array, predictions_array, db_name)

        return mse, movement_accuracy
    if action == 'evaluate':
        conn = process_data_helper.connect_to_db(db_name)
        dataset_id = "processed_data"
        retrieved_dataset = process_data_helper.retrieve_dataset_from_db(conn, dataset_id)
        conn.close()

        training_set, testing_set = split_dataframe(retrieved_dataset)

        dropped_columns = []

        date_column = 'date'
        target_name = 'sales_data_current'

        scaler_name = 'scaler'

        conn = process_data_helper.connect_to_db(db_name)

        model_name = 'MP_model'
        experiment_id = "Experiment123"

        # Check if the model already exists in the database
        model_state_dict = process_data_helper.retrieve_model_from_db(conn, model_name=model_name)

        # Step 4: Define and train the model
        input_size = 15
        hidden_size = 64
        dropout_rate = 0.2

        model = SalesPredictionMLP(input_size, hidden_size, dropout_rate)

        model.load_state_dict(model_state_dict)

        scaler = process_data_helper.load_scaler_from_db(conn, scaler_name)
        # Corrected to use test_dataset for preparation
        X_test_tensor, actuals_array = prepare_data_for_evaluation(testing_set, date_column, target_name, scaler, dropped_columns)

        model.eval()
        with torch.no_grad():
            predictions = model(X_test_tensor)

        mse = mean_squared_error(actuals_array, predictions.numpy())

        # Assuming predictions and actuals have been calculated
        predictions_array = predictions.squeeze().numpy()

        movement_accuracy = plot_predictions_vs_actuals(actuals_array, predictions_array, db_name)

        return mse, movement_accuracy

    if action == 'predict-next':
        conn = process_data_helper.connect_to_db(db_name)
        dataset_id = "processed_data"
        retrieved_dataset = process_data_helper.retrieve_dataset_from_db(conn, dataset_id)
        conn.close()

        input_size=15
        lags = 7

        conn = process_data_helper.connect_to_db(db_name)

        model_name = 'MP_model'

        # Check if the model already exists in the database
        model_state_dict = process_data_helper.retrieve_model_from_db(conn, model_name=model_name)

        # Define the model
        model = SalesPredictionMLP(input_size=input_size, hidden_size=64, dropout_rate=0.2)
        # Load the model
        model.load_state_dict(model_state_dict)
        model.eval()  # Set model to evaluation mode

        scaler_name = 'scaler'

        scaler = process_data_helper.load_scaler_from_db(conn, scaler_name)

        # Ensure the date column is treated as datetime
        retrieved_dataset['date'] = pd.to_datetime(retrieved_dataset['date'])
        # Assuming 'retrieved_dataset' is your DataFrame and it has a 'date' column and a 'sales' column
        last_known_data = retrieved_dataset.iloc[-1]

        number_of_days_int = 1  # Number of days into the future you want to predict
        start_date = pd.to_datetime(last_known_data['date']) + timedelta(days=1)

        # Initialize the forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': pd.date_range(start=last_known_data['date'] + pd.Timedelta(days=1), periods=number_of_days_int),
        })

        # Initialize an empty list for additional features
        FEATURES = []

        target_name = 'sales_data_current'

        # Define explicitly excluded columns
        excluded_columns = ['date', 'sales_data_current', 'Actual Movement', 'sales_data_percentage_change']

        # Loop over each column in the dataset
        for column in retrieved_dataset.columns:
            # Check if the column is not in the excluded list and is numeric
            if column not in excluded_columns and pd.api.types.is_numeric_dtype(retrieved_dataset[column]):
                # Append the column to the FEATURES list
                FEATURES.append(column)

        predicted_sales = []
        for i in range(number_of_days_int):
            # Set lag features based on available actuals and past predictions
            for lag in range(1, lags + 1):
                if i >= lag:
                    # Use past predictions for lags
                    forecast_df.loc[i, f'{target_name}_lag_{lag}'] = predicted_sales[-lag]
                else:
                    # Use actuals from the historical data for initial lags
                    forecast_df.loc[i, f'{target_name}_lag_{lag}'] = retrieved_dataset[target_name].iloc[-lag]

            # Generate additional features based on the date
            forecast_df.loc[i, 'dayofweek'] = forecast_df.loc[i, 'date'].dayofweek
            forecast_df.loc[i, 'quarter'] = forecast_df.loc[i, 'date'].quarter
            forecast_df.loc[i, 'month'] = forecast_df.loc[i, 'date'].month
            forecast_df.loc[i, 'year'] = forecast_df.loc[i, 'date'].year
            forecast_df.loc[i, 'dayofyear'] = forecast_df.loc[i, 'date'].dayofyear

            if i > 0:
                # For subsequent predictions, use the last prediction
                forecast_df.loc[i, 'sales_data_previous'] = predicted_sales[-1]
                forecast_df.loc[i, 'previous_sales_data_percentage_change'] = \
                    (predicted_sales[-1] - predicted_sales[-2]) / predicted_sales[-2] * 100 if len(predicted_sales) > 1 and predicted_sales[-2] != 0 else 0

                forecast_df.loc[i, 'sales_data_percentage_change'] = \
                    (predicted_sales[-1] - forecast_df.loc[i, 'sales_data_previous']) / forecast_df.loc[i, 'sales_data_previous'] * 100 if forecast_df.loc[i, 'sales_data_previous'] != 0 else 0

                # Determine movements based on the predicted and previous sales data
                forecast_df.loc[i, 'Actual Movement'] = 1 if predicted_sales[-1] > forecast_df.loc[i, 'sales_data_previous'] else \
                    (-1 if predicted_sales[-1] < forecast_df.loc[i, 'sales_data_previous'] else 0)
                forecast_df.loc[i, 'Previous Movement'] = forecast_df.loc[i-1, 'Actual Movement'] if i > 0 else 0
            else:
                # For the first prediction, use the last known value
                forecast_df.loc[i, 'sales_data_previous'] = last_known_data['sales_data_previous']
                forecast_df.loc[i, 'previous_sales_data_percentage_change'] = last_known_data['previous_sales_data_percentage_change']
                forecast_df.loc[i, 'sales_data_percentage_change'] = last_known_data['sales_data_percentage_change']
                forecast_df.loc[i, 'Actual Movement'] = last_known_data['Actual Movement']
                forecast_df.loc[i, 'Previous Movement'] = last_known_data['Previous Movement']

        # First, prepare the features for the last known day to predict its sales
        last_known_features = retrieved_dataset.drop([target_name] + excluded_columns, axis=1).iloc[-1:]
        last_known_features_scaled = scaler.transform(last_known_features)
        with torch.no_grad():
            last_day_prediction = model(torch.tensor(last_known_features_scaled).float())
        last_day_predicted_sales = last_day_prediction.item()

        # Prepare the features for the next day's prediction
        next_day_features = forecast_df[FEATURES].iloc[[0]]  # Since we're only predicting for one next day
        next_day_features_scaled = scaler.transform(next_day_features)
        with torch.no_grad():
            next_day_prediction = model(torch.tensor(next_day_features_scaled).float())
        next_day_predicted_sales = next_day_prediction.item()

        predicted_movement = "Increase" if next_day_predicted_sales > last_day_predicted_sales else "Decrease" if next_day_predicted_sales < last_day_predicted_sales else "Stable"

        # Output the predictions and the predicted movement
        print(f"Predicted sales for the last day ({last_known_data['date'].strftime('%Y-%m-%d')}): {last_day_predicted_sales}")
        print(f"Predicted sales for the next day ({start_date.strftime('%Y-%m-%d')}): {next_day_predicted_sales}")
        print(f"Predicted movement: {predicted_movement}")

        return last_day_predicted_sales, next_day_predicted_sales, predicted_movement
    if action == 'prophet':
        train_df, test_df = split_data_for_prophet_evaluation(retrieved_dataset, 0.8)
        model_prophet = Prophet(
            growth='linear',  # or 'logistic' if you expect a saturating growth
            daily_seasonality=True,  # Enable daily seasonality since you have daily data
            weekly_seasonality=True,  # Disable weekly seasonality if it's not relevant
            yearly_seasonality=True,  # Disable yearly seasonality if it's not relevant
            seasonality_mode='multiplicative',  # Choose 'additive' or 'multiplicative' based on your data
            changepoint_prior_scale=0.05,  # Adjust this to make the trend more flexible
            seasonality_prior_scale=10.0,  # Adjust this for the flexibility of the seasonality
            holidays_prior_scale=10.0,  # Adjust this for the effect of holidays
            changepoint_range=0.8,  # Proportion of history considered for changepoint detection
            n_changepoints=25,  # Number of potential changepoints to include
            interval_width=0.95,  # Width of the uncertainty intervals
        )
        # Add additional regressors
        # Initialize an empty list for additional features
        additional_features  = []

        # Define explicitly excluded columns
        excluded_columns = ['date', 'sales_data_current', 'Actual Movement', 'sales_data_percentage_change']

        # Loop over each column in the dataset
        for column in retrieved_dataset.columns:
            # Check if the column is not in the excluded list and is numeric
            if column not in excluded_columns and pd.api.types.is_numeric_dtype(retrieved_dataset[column]):
                # Append the column to the additional_features  list
                additional_features .append(column)

        # Output the FEATURES to ensure they are correctly identified
        print("Identified additional_features:", additional_features)

        for feature in additional_features:
            model_prophet.add_regressor(feature)

        model_prophet.fit(train_df)
        # Serialize the Prophet model
        serialized_model = pickle.dumps(model_prophet)
        conn = process_data_helper.connect_to_db(db_name)  # Make sure you define or replace `connect_to_db` with your actual database connection function
        model_name = "Prophet"
        experiment_id = "Experiment123"
        # Save the model to the database
        process_data_helper.save_prophet_model_to_db(conn, model_name, serialized_model, experiment_id)
        mae, rmse, mape, movement_accuracy, forecast, test_df_1 = predict_and_evaluate_model(model_prophet, test_df)
        save_forecast_plot(forecast, test_df_1, db_name)

        return mae, rmse, mape, movement_accuracy
    if action == 'prophet-evaluate':
        dataset_id = 'processed_data_for_prophet'
        conn = process_data_helper.connect_to_db(db_name)
        model_name = "Prophet"
        retrieved_dataset = process_data_helper.retrieve_dataset_from_db(conn, dataset_id)
        train_df, test_df = split_data_for_prophet_evaluation(retrieved_dataset, 0.8)
        conn = process_data_helper.connect_to_db(db_name)
        prophet_model = process_data_helper.retrieve_model_from_db_prophet_xgboost(conn, model_name)
        mae, rmse, mape, movement_accuracy, forecast, test_df_1 = predict_and_evaluate_model(prophet_model, test_df)
        return mae, rmse, mape, movement_accuracy
    if action == 'prophet-predict':
        dataset_id = 'processed_data_for_prophet'
        conn = process_data_helper.connect_to_db(db_name)
        model_name = "Prophet"
        retrieved_dataset = process_data_helper.retrieve_dataset_from_db(conn, dataset_id)

        retrieved_dataset['date'] = pd.to_datetime(retrieved_dataset['date'])

        dataset_ids = process_data_helper.get_all_dataset_ids(db_name)

        # Rename columns to fit Prophet's expected column names
        retrieved_dataset = retrieved_dataset.rename(columns={'date': 'ds', 'sales_data_current': 'y'})

        # Then, find the maximum date in the 'date' column
        max_date = retrieved_dataset['ds'].max()

        # Convert the number of days from string to integer
        number_of_days_int = int(number_of_days)

        # Calculate the end date by adding number_of_days to max_date
        end_date = max_date + pd.Timedelta(days=number_of_days_int)

        # Create a date range from max_date to end_date
        future_dates = pd.date_range(start=max_date, end=end_date)

        future_df = pd.DataFrame(future_dates, columns=['ds'])

        FEATURES = ['dayofweek', 'quarter', 'month', 'dayofyear']

        # Example setup - ensure these match your actual model and dataset
        model_name = "Prophet"

        conn = process_data_helper.connect_to_db(db_name)  # Ensure this function is correctly defined

        prophet_model = process_data_helper.retrieve_model_from_db_prophet_xgboost(conn, model_name)

        last_known_data = retrieved_dataset.iloc[-1]  # The last row of your historical data

        # Initialize the forecast DataFrame
        forecast_df = pd.DataFrame({
            'ds': pd.date_range(start=last_known_data['ds'] + pd.Timedelta(days=1), periods=number_of_days_int),
        })

        lags = 7

        target_name = 'y'

        target_name_for_regressor = 'sales_data_current'

        # Iteratively forecast and calculate features
        predicted_sales = []
        for i in range(number_of_days_int):
            # Set lag features based on available actuals and past predictions
            for lag in range(1, lags + 1):
                if i >= lag:
                    # Use past predictions for lags
                    forecast_df.loc[i, f'{target_name_for_regressor}_lag_{lag}'] = predicted_sales[-lag]
                else:
                    # Use actuals from the historical data for initial lags
                    forecast_df.loc[i, f'{target_name_for_regressor}_lag_{lag}'] = retrieved_dataset[target_name].iloc[-lag]

            # Generate additional features based on the date
            forecast_df.loc[i, 'dayofweek'] = forecast_df.loc[i, 'ds'].dayofweek
            forecast_df.loc[i, 'quarter'] = forecast_df.loc[i, 'ds'].quarter
            forecast_df.loc[i, 'month'] = forecast_df.loc[i, 'ds'].month
            forecast_df.loc[i, 'year'] = forecast_df.loc[i, 'ds'].year
            forecast_df.loc[i, 'dayofyear'] = forecast_df.loc[i, 'ds'].dayofyear

            if i > 0:
                # For subsequent predictions, use the last prediction
                forecast_df.loc[i, 'sales_data_previous'] = predicted_sales[-1]
                forecast_df.loc[i, 'previous_sales_data_percentage_change'] = \
                    (predicted_sales[-1] - predicted_sales[-2]) / predicted_sales[-2] * 100 if len(predicted_sales) > 1 and predicted_sales[-2] != 0 else 0

                forecast_df.loc[i, 'sales_data_percentage_change'] = \
                    (predicted_sales[-1] - forecast_df.loc[i, 'sales_data_previous']) / forecast_df.loc[i, 'sales_data_previous'] * 100 if forecast_df.loc[i, 'sales_data_previous'] != 0 else 0

                # Determine movements based on the predicted and previous sales data
                forecast_df.loc[i, 'Actual Movement'] = 1 if predicted_sales[-1] > forecast_df.loc[i, 'sales_data_previous'] else \
                    (-1 if predicted_sales[-1] < forecast_df.loc[i, 'sales_data_previous'] else 0)
                forecast_df.loc[i, 'Previous Movement'] = forecast_df.loc[i-1, 'Actual Movement'] if i > 0 else 0
            else:
                # For the first prediction, use the last known value
                forecast_df.loc[i, 'sales_data_previous'] = last_known_data['sales_data_previous']
                forecast_df.loc[i, 'previous_sales_data_percentage_change'] = last_known_data['previous_sales_data_percentage_change']
                forecast_df.loc[i, 'sales_data_percentage_change'] = last_known_data['sales_data_percentage_change']
                forecast_df.loc[i, 'Actual Movement'] = last_known_data['Actual Movement']
                forecast_df.loc[i, 'Previous Movement'] = last_known_data['Previous Movement']



            # Make a prediction for the current day in forecast_df
            current_prediction = prophet_model.predict(forecast_df.iloc[[i]])
            # Append the predicted 'yhat' value to predicted_sales
            predicted_sales.append(current_prediction['yhat'].iloc[0])

        # After the loop, forecast_df contains all the predictions and derived features


        # Ensure forecast_df has the correct index setup if it's not already indexed by 'ds'
        forecast_df = forecast_df.set_index('ds')

        # After the loop, you can directly assign the list of predicted_sales to forecast_df
        forecast_df['pred'] = predicted_sales

        save_predictions_plot(forecast_df, db_name, action)  # Ensure this function is correctly defined

        # Assuming predictions are already made and 'date' is the index
        predictions_with_dates = forecast_df[['pred']]

        # If you need to reset the index to work with 'date' as a column
        predictions_with_dates.reset_index(inplace=True)

        save_predictions_to_database(predictions_with_dates, db_name)

        # Convert to a dictionary for easy JSON serialization, if needed
        predictions_dict = predictions_with_dates.to_dict(orient='records')

        return predictions_dict, dataset_ids
    if action == 'xgboost':
        train_df, test_df = split_data_for_xgboost_evaluation(retrieved_dataset, 0.8)
        # Initialize an empty list for additional features
        FEATURES = []

        # Define explicitly excluded columns
        excluded_columns = ['date', 'sales_data_current', 'Actual Movement', 'sales_data_percentage_change']

        # Loop over each column in the dataset
        for column in retrieved_dataset.columns:
            # Check if the column is not in the excluded list and is numeric
            if column not in excluded_columns and pd.api.types.is_numeric_dtype(retrieved_dataset[column]):
                # Append the column to the FEATURES list
                FEATURES.append(column)

        # Output the FEATURES to ensure they are correctly identified
        print("Identified FEATURES:", FEATURES)

        TARGET = 'sales_data_current'
        X_train = train_df[FEATURES]
        y_train = train_df[TARGET]

        X_test = test_df[FEATURES]
        y_test = test_df[TARGET]

        reg = xgb.XGBRegressor(n_estimators=10000, early_stopping_rounds=100,
                                learning_rate=0.01)
        reg.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=100,)

        serialized_model = pickle.dumps(reg)
        conn = process_data_helper.connect_to_db(db_name)  # Make sure you define or replace `connect_to_db` with your actual database connection function
        model_name = "xgboost"
        experiment_id = "Experiment123"
        # Save the model to the database

        process_data_helper.save_prophet_model_to_db(conn, model_name, serialized_model, experiment_id)

        feature_importance_xgboost(reg, db_name)

        test_df['prediction'] = reg.predict(X_test)

        df = retrieved_dataset.merge(test_df['prediction'], how='left', left_index=True, right_index=True)

        rmse_fixed, df_updated, movement_accuracy = evaluate_predictions_xgboost(df)

        save_predictions_plot(df_updated, db_name, action)

        return rmse_fixed, movement_accuracy
    if action == 'xgboost-evaluate':
        # Assuming 'processed_data' is the ID for the latest dataset
        dataset_id = "processed_data_for_xgboost"
        conn = process_data_helper.connect_to_db(db_name)
        # Retrieve the latest dataset for prediction
        retrieved_dataset = process_data_helper.retrieve_dataset_from_db(conn, dataset_id)
        train_df, test_df = split_data_for_xgboost_evaluation(retrieved_dataset, 0.8)
        # Initialize an empty list for additional features
        FEATURES = []

        # Define explicitly excluded columns
        excluded_columns = ['date', 'sales_data_current', 'Actual Movement', 'sales_data_percentage_change']

        # Loop over each column in the dataset
        for column in retrieved_dataset.columns:
            # Check if the column is not in the excluded list and is numeric
            if column not in excluded_columns and pd.api.types.is_numeric_dtype(retrieved_dataset[column]):
                # Append the column to the FEATURES list
                FEATURES.append(column)

        TARGET = 'sales_data_current'

        X_test = test_df[FEATURES]
        y_test = test_df[TARGET]

        model_name = "xgboost"

        conn = process_data_helper.connect_to_db(db_name)  # Make sure you define or replace `connect_to_db` with your actual database connection function

        xgboost_model = process_data_helper.retrieve_model_from_db_prophet_xgboost(conn, model_name)

        test_df['prediction'] = xgboost_model.predict(X_test)

        df = retrieved_dataset.merge(test_df['prediction'], how='left', left_index=True, right_index=True)

        rmse_fixed, df_updated, movement_accuracy = evaluate_predictions_xgboost(df)

        return rmse_fixed, movement_accuracy
    if action == 'xgboost-prediction':
        # Assuming 'processed_data' is the ID for the latest dataset
        dataset_id = "processed_data_for_xgboost"
        conn = process_data_helper.connect_to_db(db_name)
        # Retrieve the latest dataset for prediction
        retrieved_dataset = process_data_helper.retrieve_dataset_from_db(conn, dataset_id)

        retrieved_dataset['date'] = pd.to_datetime(retrieved_dataset['date'])

        # Then, find the maximum date in the 'date' column
        max_date = retrieved_dataset['date'].max()

        # Convert the number of days from string to integer
        number_of_days_int = int(number_of_days)

        # Calculate the end date by adding number_of_days to max_date
        end_date = max_date + pd.Timedelta(days=number_of_days_int)

        # Create a date range from max_date to end_date
        future_dates = pd.date_range(start=max_date, end=end_date)

        future_df = pd.DataFrame(index=future_dates)

        FEATURES = ['dayofweek', 'quarter', 'month', 'dayofyear']

        # Example setup - ensure these match your actual model and dataset
        model_name = "xgboost"

        conn = process_data_helper.connect_to_db(db_name)  # Ensure this function is correctly defined

        xgboost_model = process_data_helper.retrieve_model_from_db_prophet_xgboost(conn, model_name)

        last_known_data = retrieved_dataset.iloc[-1]  # The last row of your historical data

        # Initialize the forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': pd.date_range(start=last_known_data['date'] + pd.Timedelta(days=1), periods=number_of_days_int),
        })

        lags = 7

        target_name = 'sales_data_current'

        # Initialize an empty list for additional features
        FEATURES = []

        # Define explicitly excluded columns
        excluded_columns = ['date', 'sales_data_current', 'Actual Movement', 'sales_data_percentage_change']

        # Loop over each column in the dataset
        for column in retrieved_dataset.columns:
            # Check if the column is not in the excluded list and is numeric
            if column not in excluded_columns and pd.api.types.is_numeric_dtype(retrieved_dataset[column]):
                # Append the column to the FEATURES list
                FEATURES.append(column)

        # Iteratively forecast and calculate features
        predicted_sales = []
        for i in range(number_of_days_int):
            # Set lag features based on available actuals and past predictions
            for lag in range(1, lags + 1):
                if i >= lag:
                    # Use past predictions for lags
                    forecast_df.loc[i, f'{target_name}_lag_{lag}'] = predicted_sales[-lag]
                else:
                    # Use actuals from the historical data for initial lags
                    forecast_df.loc[i, f'{target_name}_lag_{lag}'] = retrieved_dataset[target_name].iloc[-lag]

            # Generate additional features based on the date
            forecast_df.loc[i, 'dayofweek'] = forecast_df.loc[i, 'date'].dayofweek
            forecast_df.loc[i, 'quarter'] = forecast_df.loc[i, 'date'].quarter
            forecast_df.loc[i, 'month'] = forecast_df.loc[i, 'date'].month
            forecast_df.loc[i, 'year'] = forecast_df.loc[i, 'date'].year
            forecast_df.loc[i, 'dayofyear'] = forecast_df.loc[i, 'date'].dayofyear

            if i > 0:
                # For subsequent predictions, use the last prediction
                forecast_df.loc[i, 'sales_data_previous'] = predicted_sales[-1]
                forecast_df.loc[i, 'previous_sales_data_percentage_change'] = \
                    (predicted_sales[-1] - predicted_sales[-2]) / predicted_sales[-2] * 100 if len(predicted_sales) > 1 and predicted_sales[-2] != 0 else 0

                forecast_df.loc[i, 'sales_data_percentage_change'] = \
                    (predicted_sales[-1] - forecast_df.loc[i, 'sales_data_previous']) / forecast_df.loc[i, 'sales_data_previous'] * 100 if forecast_df.loc[i, 'sales_data_previous'] != 0 else 0

                # Determine movements based on the predicted and previous sales data
                forecast_df.loc[i, 'Actual Movement'] = 1 if predicted_sales[-1] > forecast_df.loc[i, 'sales_data_previous'] else \
                    (-1 if predicted_sales[-1] < forecast_df.loc[i, 'sales_data_previous'] else 0)
                forecast_df.loc[i, 'Previous Movement'] = forecast_df.loc[i-1, 'Actual Movement'] if i > 0 else 0
            else:
                # For the first prediction, use the last known value
                forecast_df.loc[i, 'sales_data_previous'] = last_known_data['sales_data_previous']
                forecast_df.loc[i, 'previous_sales_data_percentage_change'] = last_known_data['previous_sales_data_percentage_change']
                forecast_df.loc[i, 'sales_data_percentage_change'] = last_known_data['sales_data_percentage_change']
                forecast_df.loc[i, 'Actual Movement'] = last_known_data['Actual Movement']
                forecast_df.loc[i, 'Previous Movement'] = last_known_data['Previous Movement']



            # Select only the columns specified in FEATURES for the prediction
            features_for_prediction = forecast_df[FEATURES].iloc[[i]]

            # Make a prediction for the current day using only the selected features
            current_prediction = xgboost_model.predict(features_for_prediction)

            # Assuming current_prediction returns an array, get the first element as the prediction
            predicted_value = current_prediction[0]

            print(current_prediction)

            # Append the predicted value to predicted_sales
            predicted_sales.append(predicted_value)

        # After the loop, forecast_df contains all the predictions and derived features


        # Ensure forecast_df has the correct index setup if it's not already indexed by 'ds'
        forecast_df = forecast_df.set_index('date')

        # After the loop, you can directly assign the list of predicted_sales to forecast_df
        forecast_df['pred'] = predicted_sales

        save_predictions_plot(forecast_df, db_name, action)  # Ensure this function is correctly defined

        # Assuming predictions are already made and 'date' is the index
        predictions_with_dates = forecast_df[['pred']]

        # If you need to reset the index to work with 'date' as a column
        predictions_with_dates.reset_index(inplace=True)

        save_predictions_to_database(predictions_with_dates, db_name)

        # Convert to a dictionary for easy JSON serialization, if needed
        predictions_dict = predictions_with_dates.to_dict(orient='records')

        return predictions_dict
    if action == 'ensemble':
        # Initialize a dictionary to hold all predictions
        ensemble_predictions = {
            'XGBoost': [],
            'Prophet': [],
            'LSTM': []
        }

        # Initialize arrays to hold the separated db names based on their suffix
        xgboost_dbs = []
        prophet_dbs = []
        lstm_dbs = []

        # Define the suffixes to check for in each db name
        suffixes = {
            ' - XGBoost DB': xgboost_dbs,
            ' - Prophet DB': prophet_dbs,
            ' - LSTM DB': lstm_dbs
        }

        # Loop through the db names
        for name in db_name:
            # Check against each suffix
            for suffix, db_array in suffixes.items():
                if suffix in name:
                    # Add the db name to the corresponding array if the suffix is found
                    db_array.append(name)

        # Process each XGBoost DB
        for path in xgboost_dbs:
            LSTM_predictions = None
            predictions = []  # This will hold tuples of (date, prediction)
            dataset_id = "processed_data_for_xgboost"
            conn = process_data_helper.connect_to_db(path)
            # Retrieve the latest dataset for prediction
            retrieved_dataset = process_data_helper.retrieve_dataset_from_db(conn, dataset_id)
            dataset_ids = process_data_helper.get_all_dataset_ids(path)
            train_df, test_df = split_data_for_xgboost_evaluation(retrieved_dataset, 0.8)
            # Initialize an empty list for additional features
            FEATURES = []

            # Define explicitly excluded columns
            excluded_columns = ['date', 'sales_data_current', 'Actual Movement', 'sales_data_percentage_change']

            # Loop over each column in the dataset
            for column in retrieved_dataset.columns:
                # Check if the column is not in the excluded list and is numeric
                if column not in excluded_columns and pd.api.types.is_numeric_dtype(retrieved_dataset[column]):
                    # Append the column to the FEATURES list
                    FEATURES.append(column)

            TARGET = 'sales_data_current'

            X_test = test_df[FEATURES]
            y_test = test_df[TARGET]

            model_name = "xgboost"

            conn = process_data_helper.connect_to_db(path)  # Make sure you define or replace `connect_to_db` with your actual database connection function

            xgboost_model = process_data_helper.retrieve_model_from_db_prophet_xgboost(conn, model_name)

            test_df['prediction'] = xgboost_model.predict(X_test)

            df = retrieved_dataset.merge(test_df['prediction'], how='left', left_index=True, right_index=True)

            rmse_fixed, df_updated, movement_accuracy = evaluate_predictions_xgboost(df)

            # Assuming 'date' is your datetime column and you've merged such that
            # each prediction aligns with its corresponding date in 'df'
            for _, row in df.iterrows():
                date = row['date']
                prediction = row['prediction']
                # Add the (date, prediction) tuple to the predictions list
                predictions.append((date, prediction))

            # Store these predictions in your ensemble predictions structure
            ensemble_predictions['XGBoost'].append(predictions)

            xgboost_predictions = ensemble_predictions['XGBoost']

        # Process each Prophet DB
        for path in prophet_dbs:
            dataset_id = "processed_data_for_prophet"
            conn = process_data_helper.connect_to_db(path)
            # Retrieve the latest dataset for prediction
            retrieved_dataset = process_data_helper.retrieve_dataset_from_db(conn, dataset_id)
            dataset_id = 'processed_data_for_prophet'
            conn = process_data_helper.connect_to_db(path)
            model_name = "Prophet"
            retrieved_dataset = process_data_helper.retrieve_dataset_from_db(conn, dataset_id)
            train_df, test_df = split_data_for_prophet_evaluation(retrieved_dataset, 0.8)
            conn = process_data_helper.connect_to_db(path)
            prophet_model = process_data_helper.retrieve_model_from_db_prophet_xgboost(conn, model_name)
            mae, rmse, mape, movement_accuracy, forecast, test_df_1 = predict_and_evaluate_model(prophet_model, test_df)

            predictions = []  # This will hold tuples of (date, prediction)

            # Ensuring dates in forecast match those in actual_dates from test_df
            actual_dates = test_df['ds']
            matched_forecast = forecast.loc[forecast['ds'].isin(actual_dates)]

            # Now, matched_forecast contains only the rows where the forecast dates match the test set dates
            predictions = []  # Initialize a list to store tuples of (date, prediction)
            for index, row in matched_forecast.iterrows():
                date = row['ds']
                prediction = row['yhat']  # Assuming 'yhat' is the column with predictions
                predictions.append((date, prediction))

            # Append these matched date-prediction pairs to the ensemble_predictions under 'Prophet'
            ensemble_predictions['Prophet'].append(predictions)

            prophet_predictions = ensemble_predictions['Prophet']

        df_merged = pd.DataFrame()

        # Iterate over each model type in ensemble_predictions
        for model_type, predictions_list in ensemble_predictions.items():
            # Check if there are any predictions for the current model type
            if predictions_list:
                # Flatten the list of predictions
                flat_predictions = [item for sublist in predictions_list for item in sublist]
                # Create a DataFrame from the flattened list
                df_model = pd.DataFrame(flat_predictions, columns=['date', f'{model_type}_Prediction'])
                # Convert 'date' to datetime format for consistent merging
                df_model['date'] = pd.to_datetime(df_model['date'])

                # Drop duplicate dates within the same model's predictions
                df_model = df_model.drop_duplicates(subset=['date'], keep='first')

                if df_merged.empty:
                    # If df_merged is empty, this is the first non-empty model we're adding
                    df_merged = df_model
                else:
                    # Merge with the existing df_merged on 'date', using outer join to keep all dates
                    df_merged = pd.merge(df_merged, df_model, on='date', how='outer')

        # Sort the merged DataFrame by date to ensure chronological order
        df_merged.sort_values('date', inplace=True)

        sales_data_df = retrieved_dataset[['date', 'sales_data_current']].drop_duplicates(subset=['date'])

        # Convert 'date' to datetime format for consistent merging
        sales_data_df['date'] = pd.to_datetime(sales_data_df['date'])

        # Merge sales data with df_merged on 'date'
        df_merged_with_sales = pd.merge(df_merged, sales_data_df, on='date', how='left')

        df_merged_with_sales.sort_values('date', inplace=True)

        split_point = int(len(df_merged_with_sales) * 0.8)

        # Prepare the features and target variable
        X = df_merged_with_sales[['XGBoost_Prediction', 'Prophet_Prediction']]
        y = df_merged_with_sales['sales_data_current']

        # Split the data
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]

        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)

        # Make predictions on the test set
        test_predictions = linear_model.predict(X_test)

        # Assuming the DataFrame is still sorted by date and split by index, we add predictions to the test portion.
        # We'll create a temporary column that we'll later fill with predictions for the test set portion
        df_merged_with_sales['Linear_Regression_Prediction'] = None

        # Now, assign the predictions to the corresponding rows in the DataFrame
        # We use the split_point to determine where the test set starts
        df_merged_with_sales.iloc[split_point:, df_merged_with_sales.columns.get_loc('Linear_Regression_Prediction')] = test_predictions

        # Optionally, convert the predictions column to the correct dtype if necessary, e.g., float
        df_merged_with_sales['prediction'] = pd.to_numeric(df_merged_with_sales['Linear_Regression_Prediction'], errors='coerce')

        # Creating a new DataFrame that contains only rows with no missing values across any column. Now, df_complete_cases contains only the rows where every column has a value.
        df_complete_cases = df_merged_with_sales.dropna()

        df_final = df_complete_cases.drop(columns=['XGBoost_Prediction', 'Prophet_Prediction'])

        rmse = np.sqrt(mean_squared_error(df_final['sales_data_current'], df_final['prediction']))  # Use np.sqrt to get RMSE

        # Calculate Actual Movement
        df_final['Actual Movement'] = np.sign(df_final['sales_data_current'].diff().fillna(0))

        # Calculate Predicted Movement
        df_final['Predicted Movement'] = np.sign(df_final['prediction'].diff().fillna(0))

        # Ensure the first value of 'Predicted Movement' isn't NaN due to the diff operation
        df_final['Predicted Movement'] = df_final['Predicted Movement'].fillna(0)

        # Map predicted movements to the corresponding dates in test_df
        df_final['Predicted Movement'] = df_final['date'].map(df_final.set_index('date')['Predicted Movement']).fillna(0)

        # Calculate movement prediction accuracy
        movement_accuracy = accuracy_score(df_final['Actual Movement'], df_final['Predicted Movement']) * 100  # Convert to percentage

        rmse_fixed = f"{rmse:.2f}"

        all_paths = lstm_dbs + prophet_dbs + xgboost_dbs

        for path in all_paths:
            save_predictions_plot(df_final, path, action)

            conn = process_data_helper.connect_to_db(path)

            serialized_model = pickle.dumps(linear_model)

            model_name = "linear_model"
            experiment_id = "Experiment123"
            # Save the model to the database
            process_data_helper.save_prophet_model_to_db(conn, model_name, serialized_model, experiment_id)

        return xgboost_dbs, prophet_dbs, lstm_dbs, db_name, xgboost_predictions, prophet_predictions, LSTM_predictions, rmse_fixed, movement_accuracy, dataset_ids
    if action == 'ensemble_prediction':
        LSTM_predictions = None
        # Convert the number of days from string to integer
        number_of_days_int = int(number_of_days)
        # Initialize a dictionary to hold all predictions
        ensemble_predictions_future = {
            'XGBoost': [],
            'Prophet': [],
            'LSTM': []
        }

        # Initialize arrays to hold the separated db names based on their suffix
        xgboost_dbs_future = []
        prophet_dbs_future = []
        lstm_dbs_future = []

        # Define the suffixes to check for in each db name
        suffixes = {
            ' - XGBoost DB': xgboost_dbs_future,
            ' - Prophet DB': prophet_dbs_future,
            ' - LSTM DB': lstm_dbs_future
        }

        # Loop through the db names
        for name in db_name:
            # Check against each suffix
            for suffix, db_array in suffixes.items():
                if suffix in name:
                    # Add the db name to the corresponding array if the suffix is found
                    db_array.append(name)

        for path in xgboost_dbs_future:
            predictions = []

            # Assuming 'processed_data' is the ID for the latest dataset
            dataset_id = "processed_data_for_xgboost"
            conn = process_data_helper.connect_to_db(path)
            # Retrieve the latest dataset for prediction
            retrieved_dataset = process_data_helper.retrieve_dataset_from_db(conn, dataset_id)

            retrieved_dataset['date'] = pd.to_datetime(retrieved_dataset['date'])

            # Then, find the maximum date in the 'date' column
            max_date = retrieved_dataset['date'].max()

            # Calculate the end date by adding number_of_days to max_date
            end_date = max_date + pd.Timedelta(days=number_of_days_int)

            # Create a date range from max_date to end_date
            future_dates = pd.date_range(start=max_date, end=end_date)

            future_df = pd.DataFrame(index=future_dates)

            FEATURES = ['dayofweek', 'quarter', 'month', 'dayofyear']

            model_name = "xgboost"

            conn = process_data_helper.connect_to_db(path)  # Make sure you define or replace `connect_to_db` with your actual database connection function

            xgboost_model = process_data_helper.retrieve_model_from_db_prophet_xgboost(conn, model_name)

            last_known_data = retrieved_dataset.iloc[-1]  # The last row of your historical data

            # Initialize the forecast DataFrame
            forecast_df = pd.DataFrame({
                'date': pd.date_range(start=last_known_data['date'] + pd.Timedelta(days=1), periods=number_of_days_int),
            })

            lags = 7

            target_name = 'sales_data_current'

            # Initialize an empty list for additional features
            FEATURES = []

            # Define explicitly excluded columns
            excluded_columns = ['date', 'sales_data_current', 'Actual Movement', 'sales_data_percentage_change']

            # Loop over each column in the dataset
            for column in retrieved_dataset.columns:
                # Check if the column is not in the excluded list and is numeric
                if column not in excluded_columns and pd.api.types.is_numeric_dtype(retrieved_dataset[column]):
                    # Append the column to the FEATURES list
                    FEATURES.append(column)

            # Iteratively forecast and calculate features
            predicted_sales = []
            for i in range(number_of_days_int):
                # Set lag features based on available actuals and past predictions
                for lag in range(1, lags + 1):
                    if i >= lag:
                        # Use past predictions for lags
                        forecast_df.loc[i, f'{target_name}_lag_{lag}'] = predicted_sales[-lag]
                    else:
                        # Use actuals from the historical data for initial lags
                        forecast_df.loc[i, f'{target_name}_lag_{lag}'] = retrieved_dataset[target_name].iloc[-lag]

                # Generate additional features based on the date
                forecast_df.loc[i, 'dayofweek'] = forecast_df.loc[i, 'date'].dayofweek
                forecast_df.loc[i, 'quarter'] = forecast_df.loc[i, 'date'].quarter
                forecast_df.loc[i, 'month'] = forecast_df.loc[i, 'date'].month
                forecast_df.loc[i, 'year'] = forecast_df.loc[i, 'date'].year
                forecast_df.loc[i, 'dayofyear'] = forecast_df.loc[i, 'date'].dayofyear

                if i > 0:
                    # For subsequent predictions, use the last prediction
                    forecast_df.loc[i, 'sales_data_previous'] = predicted_sales[-1]
                    forecast_df.loc[i, 'previous_sales_data_percentage_change'] = \
                        (predicted_sales[-1] - predicted_sales[-2]) / predicted_sales[-2] * 100 if len(predicted_sales) > 1 and predicted_sales[-2] != 0 else 0

                    forecast_df.loc[i, 'sales_data_percentage_change'] = \
                        (predicted_sales[-1] - forecast_df.loc[i, 'sales_data_previous']) / forecast_df.loc[i, 'sales_data_previous'] * 100 if forecast_df.loc[i, 'sales_data_previous'] != 0 else 0

                    # Determine movements based on the predicted and previous sales data
                    forecast_df.loc[i, 'Actual Movement'] = 1 if predicted_sales[-1] > forecast_df.loc[i, 'sales_data_previous'] else \
                        (-1 if predicted_sales[-1] < forecast_df.loc[i, 'sales_data_previous'] else 0)
                    forecast_df.loc[i, 'Previous Movement'] = forecast_df.loc[i-1, 'Actual Movement'] if i > 0 else 0
                else:
                    # For the first prediction, use the last known value
                    forecast_df.loc[i, 'sales_data_previous'] = last_known_data['sales_data_previous']
                    forecast_df.loc[i, 'previous_sales_data_percentage_change'] = last_known_data['previous_sales_data_percentage_change']
                    forecast_df.loc[i, 'sales_data_percentage_change'] = last_known_data['sales_data_percentage_change']
                    forecast_df.loc[i, 'Actual Movement'] = last_known_data['Actual Movement']
                    forecast_df.loc[i, 'Previous Movement'] = last_known_data['Previous Movement']



                # Select only the columns specified in FEATURES for the prediction
                features_for_prediction = forecast_df[FEATURES].iloc[[i]]

                # Make a prediction for the current day using only the selected features
                current_prediction = xgboost_model.predict(features_for_prediction)

                # Assuming current_prediction returns an array, get the first element as the prediction
                predicted_value = current_prediction[0]

                print(current_prediction)

                # Append the predicted value to predicted_sales
                predicted_sales.append(predicted_value)

            # After the loop, forecast_df contains all the predictions and derived features


            # Ensure forecast_df has the correct index setup if it's not already indexed by 'ds'
            forecast_df = forecast_df.set_index('date')

            # After the loop, you can directly assign the list of predicted_sales to forecast_df
            forecast_df['pred'] = predicted_sales

            # Assuming predictions are already made and 'date' is the index
            predictions_with_dates = forecast_df[['pred']]

            # To reset the index and turn 'date' back into a column
            forecast_df.reset_index(inplace=True)
            forecast_df.rename(columns={'index': 'date'}, inplace=True)

            # Assuming 'date' is your datetime column and you've merged such that
            # each prediction aligns with its corresponding date in 'df'
            for _, row in forecast_df.iterrows():
                date = row['date']
                prediction = row['pred']
                # Add the (date, prediction) tuple to the predictions list
                predictions.append((date, prediction))

            ensemble_predictions_future['XGBoost'].append(predictions)

            xgboost_predictions = ensemble_predictions_future['XGBoost']

        for path in prophet_dbs_future:
            predictions = []
            dataset_id = 'processed_data_for_prophet'
            conn = process_data_helper.connect_to_db(path)

            retrieved_dataset = process_data_helper.retrieve_dataset_from_db(conn, dataset_id)

            retrieved_dataset['date'] = pd.to_datetime(retrieved_dataset['date'])

            # Rename columns to fit Prophet's expected column names
            retrieved_dataset = retrieved_dataset.rename(columns={'date': 'ds', 'sales_data_current': 'y'})

            # Then, find the maximum date in the 'date' column
            max_date = retrieved_dataset['ds'].max()

            # Convert the number of days from string to integer
            number_of_days_int = int(number_of_days)

            # Calculate the end date by adding number_of_days to max_date
            end_date = max_date + pd.Timedelta(days=number_of_days_int)

            # Create a date range from max_date to end_date
            future_dates = pd.date_range(start=max_date, end=end_date)

            future_df = pd.DataFrame(future_dates, columns=['ds'])

            FEATURES = ['dayofweek', 'quarter', 'month', 'dayofyear']

            # Example setup - ensure these match your actual model and dataset
            model_name = "Prophet"

            conn = process_data_helper.connect_to_db(path)  # Ensure this function is correctly defined

            prophet_model = process_data_helper.retrieve_model_from_db_prophet_xgboost(conn, model_name)

            last_known_data = retrieved_dataset.iloc[-1]  # The last row of your historical data

            # Initialize the forecast DataFrame
            forecast_df = pd.DataFrame({
                'ds': pd.date_range(start=last_known_data['ds'] + pd.Timedelta(days=1), periods=number_of_days_int),
            })

            lags = 7

            target_name = 'y'

            target_name_for_regressor = 'sales_data_current'

            # Iteratively forecast and calculate features
            predicted_sales = []
            for i in range(number_of_days_int):
                # Set lag features based on available actuals and past predictions
                for lag in range(1, lags + 1):
                    if i >= lag:
                        # Use past predictions for lags
                        forecast_df.loc[i, f'{target_name_for_regressor}_lag_{lag}'] = predicted_sales[-lag]
                    else:
                        # Use actuals from the historical data for initial lags
                        forecast_df.loc[i, f'{target_name_for_regressor}_lag_{lag}'] = retrieved_dataset[target_name].iloc[-lag]

                # Generate additional features based on the date
                forecast_df.loc[i, 'dayofweek'] = forecast_df.loc[i, 'ds'].dayofweek
                forecast_df.loc[i, 'quarter'] = forecast_df.loc[i, 'ds'].quarter
                forecast_df.loc[i, 'month'] = forecast_df.loc[i, 'ds'].month
                forecast_df.loc[i, 'year'] = forecast_df.loc[i, 'ds'].year
                forecast_df.loc[i, 'dayofyear'] = forecast_df.loc[i, 'ds'].dayofyear

                if i > 0:
                    # For subsequent predictions, use the last prediction
                    forecast_df.loc[i, 'sales_data_previous'] = predicted_sales[-1]
                    forecast_df.loc[i, 'previous_sales_data_percentage_change'] = \
                        (predicted_sales[-1] - predicted_sales[-2]) / predicted_sales[-2] * 100 if len(predicted_sales) > 1 and predicted_sales[-2] != 0 else 0

                    forecast_df.loc[i, 'sales_data_percentage_change'] = \
                        (predicted_sales[-1] - forecast_df.loc[i, 'sales_data_previous']) / forecast_df.loc[i, 'sales_data_previous'] * 100 if forecast_df.loc[i, 'sales_data_previous'] != 0 else 0

                    # Determine movements based on the predicted and previous sales data
                    forecast_df.loc[i, 'Actual Movement'] = 1 if predicted_sales[-1] > forecast_df.loc[i, 'sales_data_previous'] else \
                        (-1 if predicted_sales[-1] < forecast_df.loc[i, 'sales_data_previous'] else 0)
                    forecast_df.loc[i, 'Previous Movement'] = forecast_df.loc[i-1, 'Actual Movement'] if i > 0 else 0
                else:
                    # For the first prediction, use the last known value
                    forecast_df.loc[i, 'sales_data_previous'] = last_known_data['sales_data_previous']
                    forecast_df.loc[i, 'previous_sales_data_percentage_change'] = last_known_data['previous_sales_data_percentage_change']
                    forecast_df.loc[i, 'sales_data_percentage_change'] = last_known_data['sales_data_percentage_change']
                    forecast_df.loc[i, 'Actual Movement'] = last_known_data['Actual Movement']
                    forecast_df.loc[i, 'Previous Movement'] = last_known_data['Previous Movement']



                # Make a prediction for the current day in forecast_df
                current_prediction = prophet_model.predict(forecast_df.iloc[[i]])
                # Append the predicted 'yhat' value to predicted_sales
                predicted_sales.append(current_prediction['yhat'].iloc[0])

            # After the loop, forecast_df contains all the predictions and derived features


            # After the loop, you can directly assign the list of predicted_sales to forecast_df
            forecast_df['pred'] = predicted_sales

            # Assuming 'date' is your datetime column and you've merged such that
            # each prediction aligns with its corresponding date in 'df'
            for _, row in forecast_df.iterrows():
                date = row['ds']
                prediction = row['pred']
                # Add the (date, prediction) tuple to the predictions list
                predictions.append((date, prediction))

            ensemble_predictions_future['Prophet'].append(predictions)
            prophet_predictions = ensemble_predictions_future['Prophet']

        df_merged = pd.DataFrame()

        # Iterate over each model type in ensemble_predictions
        for model_type, predictions_list in ensemble_predictions_future.items():
            # Check if there are any predictions for the current model type
            if predictions_list:
                # Flatten the list of predictions
                flat_predictions = [item for sublist in predictions_list for item in sublist]
                # Create a DataFrame from the flattened list
                df_model = pd.DataFrame(flat_predictions, columns=['date', f'{model_type}_Prediction'])
                # Convert 'date' to datetime format for consistent merging
                df_model['date'] = pd.to_datetime(df_model['date'])

                # Drop duplicate dates within the same model's predictions
                df_model = df_model.drop_duplicates(subset=['date'], keep='first')

                if df_merged.empty:
                    # If df_merged is empty, this is the first non-empty model we're adding
                    df_merged = df_model
                else:
                    # Merge with the existing df_merged on 'date', using outer join to keep all dates
                    df_merged = pd.merge(df_merged, df_model, on='date', how='outer')

        all_paths = lstm_dbs_future + prophet_dbs_future + xgboost_dbs_future

        FEATURES = ['XGBoost_Prediction', 'Prophet_Prediction']

        for path in all_paths:
            model_name = "linear_model"

            conn = process_data_helper.connect_to_db(path)  # Ensure this function is correctly defined

            linear_model = process_data_helper.retrieve_model_from_db_prophet_xgboost(conn, model_name)

        df_merged['pred'] = linear_model.predict(df_merged[FEATURES])

        # Dropping the 'XGBoost_Prediction' and 'Prophet_Prediction' columns from df_merged
        df_merged = df_merged.drop(columns=['XGBoost_Prediction', 'Prophet_Prediction'])

        for path in all_paths:
            conn = process_data_helper.connect_to_db(path)  # Ensure this function is correctly defined
            process_data_helper.save_dataset_to_db(conn, 'save_all_predictions', 'User Provided Dataset', df_merged)

        # Set the 'date' column as the index of the DataFrame
        df_merged.set_index('date', inplace=True)

        for path in all_paths:
            save_predictions_plot(df_merged, path, action)  # Ensure this function is correctly defined

        return number_of_days_int, xgboost_predictions, prophet_predictions, LSTM_predictions


app = Flask(__name__)

app.secret_key = '98'  # Set a secret key for session management

app.logger.setLevel(logging.INFO)  # Set the logging level to INFO

file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)

results = {}

CORS(app)

# Adjust these paths as needed
base_dir = '/home/Hero98/mysite/user_data'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

app.config['UPLOAD_FOLDER_BASE'] = os.path.join(base_dir)
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Ensure base upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER_BASE'], exist_ok=True)

# Configurations for Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'gissuliman@gmail.com'  # Your email
app.config['MAIL_PASSWORD'] = 'nvpx dzpc ugeo mqrq'  # Your email account password
app.config['MAIL_DEFAULT_SENDER'] = 'gissuliman@gmail.com'  # Also your email

mail = Mail(app)

@app.route('/send-email', methods=['POST'])
def send_email():
    data = request.get_json()
    email = data.get('email')
    message_body = data.get('message')

    msg = Message("Message from Your Website", recipients=["gissuliman@gmail.com"])
    msg.body = f"Message from: {email}\n\n{message_body}"

    try:
        mail.send(msg)
        return jsonify({"message": "Email sent successfully"}), 200
    except Exception as e:
        return jsonify({"message": "Email sending failed", "error": str(e)}), 500



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Variable to store the current DB name
current_db_name = ""

# Variable to store the current path for database and uploaded file
current_path = ""

@app.route('/')
def home():
    # This will render the initial page where users choose between "Train" or "Evaluate"
    return render_template('index.html')

@app.route('/login-page')
def login_page():
    # This will render the initial page where users choose between "Train" or "Evaluate"
    return render_template('login.html')

# Ensure you have a route for '/main.html' or adjust the redirect in login success response accordingly.
@app.route('/main')
def main():
    # Make sure this user is in session or redirect to login
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('main.html')

@app.route('/upload-csv', methods=['POST'])
def upload_csv():
    # Extract userId from form data
    user_id = request.form.get('user_id')  # Retrieve the userId from FormData
    db_name = request.form.get('db_name')  # Retrieve the userId from FormData
    action = request.form.get('action')
    db_names = request.form.get('db_name')
    if not user_id:
        return jsonify({'error': 'Missing user ID'}), 400

    current_path = os.path.join(app.config['UPLOAD_FOLDER_BASE'], user_id)
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # Check and create the directory if it doesn't exist
        if not os.path.exists(current_path):
            os.makedirs(current_path, exist_ok=True)
        filename = secure_filename(file.filename)
        file_path = os.path.join(current_path, filename)
        file.save(file_path)

        user_folder = os.path.join(app.config['UPLOAD_FOLDER_BASE'], user_id)

        if action != 'ensemble' and action != 'ensemble_prediction':
            db_path = os.path.join(user_folder, f"{db_name}.db")
            process_data_helper.create_tables(db_path)
            try:
                process_data_helper.read_user_csv_and_process_data(db_path, file_path, action)
                return jsonify({'message': 'File uploaded successfully.'}), 200
            except Exception:
                process_data_helper.read_user_csv_and_process_data(db_path, file_path)
        else:
            db_names_input = request.form.get('db_name')  # This can be a string or a comma-separated list of strings
            # Assuming db_names_input for ensemble action is a comma-separated string of db names
            db_names = db_names_input.split(',')  # Split the string into a list based on comma
            # Remove any leading/trailing whitespace from each db name
            db_names = [name.strip() for name in db_names]
            db_paths = [os.path.join(user_folder, f"{name}.db") for name in db_names]

            # Process file for each db_path
            messages = []
            for db_path in db_paths:
                try:
                    # Assuming process_data_helper.read_user_csv_and_process_data is the function you use for processing
                    process_data_helper.read_user_csv_and_process_data(db_path, file_path, action)
                    messages.append(f"File uploaded and processed successfully for {db_path}.")
                except Exception as e:
                    app.logger.error(f"Failed to process file for {db_path}: {e}")
                    messages.append(f"Failed to process file for {os.path.basename(db_path)}.")

            return jsonify({'messages': messages}), 200
    else:
        return jsonify({'message': 'File not allowed'}), 400

@app.route('/list-dbs', methods=['POST'])
def list_databases():
    data = request.get_json()
    user_id = data.get('user_id')
    action = data.get('action')

    if action == 'evaluate' or action == 'predict-next':
        suffix = ' - MP DB'
    elif action == 'prophet-evaluate' or action == 'prophet-predict':
        suffix = ' - Prophet DB'
    elif action == 'xgboost-evaluate' or action == 'xgboost-prediction':
        suffix = ' - XGBoost DB'
    elif action == 'ensemble' or action == 'ensemble_prediction':
        suffixes = (' - XGBoost DB', ' - Prophet DB', ' - LSTM DB')


    user_folder = os.path.join(app.config['UPLOAD_FOLDER_BASE'], user_id)

    if action != 'ensemble' and action != 'ensemble_prediction':
        if not os.path.exists(user_folder):
            return jsonify({'error': f'User folder for {user_id} does not exist.'}), 404

        # Filter database files based on the presence of the suffix in the filename
        if suffix:
            db_files = [f for f in os.listdir(user_folder) if suffix in f and f.endswith('.db')]
        else:
            # If no suffix is provided, list all database files
            db_files = [f for f in os.listdir(user_folder) if f.endswith('.db')]

        if not db_files:
            return jsonify({'message': 'No databases found for the specified user.'}), 404

        return jsonify({'databases': db_files}), 200
    else:
        if not os.path.exists(user_folder):
            return jsonify({'error': f'User folder for {user_id} does not exist.'}), 404

        # Initialize an empty list to hold matching database files
        db_files = []

        # Check each file in the user folder for a matching suffix and .db extension
        for file in os.listdir(user_folder):
            if file.endswith('.db') and any(suffix in file for suffix in suffixes):
                db_files.append(file)

        if not db_files:
            return jsonify({'message': 'No databases found for the specified user.'}), 404

        return jsonify({'databases': db_files}), 200

@app.route('/delete-db', methods=['POST'])
def delete_database():
    data = request.get_json()
    user_id = data.get('user_id')
    db_name = data.get('db_name')

    user_folder = os.path.join(app.config['UPLOAD_FOLDER_BASE'], user_id)
    db_file_path = os.path.join(user_folder, db_name)

    if os.path.exists(db_file_path):
        try:
            os.remove(db_file_path)
            return jsonify({'success': True}), 200
        except Exception as e:
            return jsonify({'success': False, 'message': f'Failed to delete database {db_name}. Error: {str(e)}'}), 500
    else:
        return jsonify({'success': False, 'message': f'Database {db_name} not found.'}), 404


# Assuming you've set up your MySQL connection details
db_config = {
    'user': 'Hero98',
    'password': 'Ameer1fadi@3',
    'host': 'Hero98.mysql.pythonanywhere-services.com',
    'database': 'Hero98$PredictPrice',
    'raise_on_warnings': True
}

load_dotenv("/home/Hero98/mysite/.env")

# Stripe API setup (TEST)
stripe.api_key = os.getenv('STRIPE_PRIVATE_KEY')  # Set your Stripe secret key here (TEST)

price_id = os.getenv('STRIPE_PRIVATE_PRICE') # Your Stripe Price ID

@app.route('/create-checkout-session', methods=['POST'])
def create_checkout_session():
    data = request.get_json()
    user_id = data.get('user_id')  # Retrieve the user_id sent from the frontend

    if not user_id:
        return jsonify({'error': 'User ID is missing'}), 400

    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[
                {
                    'price': price_id,
                    'quantity': 1,
                },
            ],
            mode='subscription',
            success_url=request.host_url + 'success?session_id={CHECKOUT_SESSION_ID}',
            cancel_url=request.host_url + 'cancel',
            client_reference_id=user_id,  # Pass the user_id to Stripe
        )
        return jsonify({'url': checkout_session.url})
    except Exception as e:
        return jsonify({'error': str(e)}), 403


@app.route('/success')
def success():
    session_id = request.args.get('session_id')
    stripe_session = stripe.checkout.Session.retrieve(session_id)
    user_id = stripe_session.client_reference_id  # Assuming you've set this in the checkout session

    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(buffered=True)

    subscription_date = datetime.now()
    try:
        # Update has_paid to True (1 in MySQL) for the specified user and set the subscription start date
        cursor.execute("""
            UPDATE users
            SET has_paid = %s,
                start_date_subscription = %s,
                subscription_status = 'active'
            WHERE username = %s
        """, (True, subscription_date, user_id))
        conn.commit()
    except Exception as e:
        print(e)  # Log the error or handle it as per your application's error handling policy
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

    # Redirect the user to the login page
    return redirect('/login-page')

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()  # Get data as JSON
    username = data['username']
    password = data['password']

    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(buffered=True)

    # Check if the username already exists
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    user_exists = cursor.fetchone()

    if user_exists:
        cursor.close()
        conn.close()
        return json.dumps({'success': False, 'message': 'Username already exists'}), 400, {'ContentType': 'application/json'}

    hashed_password = generate_password_hash(password, method='sha256')
    try:
        # Inside the try block of the signup function after hashing the password
        signup_date = datetime.now()
        cursor.execute("INSERT INTO users (username, password, start_date, has_paid) VALUES (%s, %s, %s, %s)",
                       (username, hashed_password, signup_date, False))
        conn.commit()
        message = 'User successfully registered. Your 7-day free trial has started. Please log in.'
        status_code = 200
    except mysql.connector.Error as err:
        message = 'Other database error'
        status_code = 400
        print(err)  # For debugging purposes
    finally:
        cursor.close()
        conn.close()

    return json.dumps({'success': status_code == 200, 'message': message}), status_code, {'ContentType': 'application/json'}

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT password, start_date, has_paid, subscription_end_date, subscription_status
            FROM users WHERE username = %s
        """, (username,))
        record = cursor.fetchone()

        if record:
            user_password, start_date, has_paid, subscription_end_date, subscription_status = record

            if not check_password_hash(user_password, password):
                return jsonify({'success': False, 'message': 'Username or password is incorrect'}), 401

            # Convert datetime to date for comparison, if start_date is not None
            start_date = start_date.date() if start_date else None

            if has_paid:
                if subscription_status:
                  # Check if there's an end date and if it's in the future
                    if subscription_end_date and datetime.now() <= subscription_end_date:
                        session['user'] = username
                        return jsonify({'success': True, 'message': 'Login successful'}), 200
                    elif not subscription_end_date:
                        # If there's no end date but the status is active, grant access
                        session['user'] = username
                        return jsonify({'success': True, 'message': 'Login successful'}), 200
                    else:
                        return jsonify({'success': False, 'message': 'Subscription expired. Please renew to continue.'}), 403
                else:
                    # User has paid but subscription_status is None
                    session['user'] = username
                    return jsonify({'success': True, 'message': 'Login successful. Please update your subscription details.'}), 200
            else:
                # Handle free trial users
                trial_end_date = start_date + timedelta(days=7) if start_date else datetime.now().date() + timedelta(days=7)
                if datetime.now().date() <= trial_end_date:
                    session['user'] = username
                    return jsonify({'success': True, 'message': 'Login successful'}), 200
                else:
                    return jsonify({'success': False, 'message': 'Free trial ended. Please subscribe to continue.'}), 403
        else:
            return jsonify({'success': False, 'message': 'Username or password is incorrect'}), 401
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals() and conn.is_connected():
            conn.close()

@app.route('/reset-password', methods=['POST'])
def reset_password():
    data = request.get_json()
    email = data.get('email')
    new_password = data.get('new_password')

    if not email or not new_password:
        return jsonify({'success': False, 'message': 'Email and new password are required.'}), 400

    # Connect to your database
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(buffered=True)

    # Check if the email exists in the database
    cursor.execute("SELECT * FROM users WHERE username = %s", (email,))
    user = cursor.fetchone()

    if not user:
        cursor.close()
        conn.close()
        return jsonify({'success': False, 'message': 'Email not found.'}), 404

    # Update the user's password
    hashed_password = generate_password_hash(new_password, method='sha256')
    cursor.execute("UPDATE users SET password = %s WHERE username = %s", (hashed_password, email))
    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({'success': True, 'message': 'Password has been reset successfully.'}), 200

@app.route('/cancel-subscription', methods=['POST'])
def cancel_subscription():
    data = request.get_json()
    username = data.get('username')  # Ensure this matches the email used in Stripe

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(buffered=True)

        # Retrieve the Stripe customer ID(s) associated with the email
        customers = stripe.Customer.list(email=username)
        if not customers.data:
            return jsonify({'success': False, 'message': 'No Stripe customer found with this email.'}), 404

        # Assuming you want to cancel subscriptions for the first customer object
        customer_id = customers.data[0].id

        # Fetch active subscriptions for the customer
        subscriptions = stripe.Subscription.list(customer=customer_id, status='active')

        if subscriptions.data:
            subscription = subscriptions.data[0]

            # Mark the subscription to cancel at the end of the billing period
            stripe.Subscription.modify(subscription.id, cancel_at_period_end=True)

            # Fetch the subscription again to get updated details
            updated_subscription = stripe.Subscription.retrieve(subscription.id)
            end_date = datetime.fromtimestamp(updated_subscription.current_period_end)

            # Format end_date for MySQL
            formatted_end_date = end_date.strftime('%Y-%m-%d %H:%M:%S')

            # Update the subscription_end_date and subscription_status in your database
            update_query = """
                UPDATE users
                SET subscription_end_date = %s, subscription_status = 'cancelled'
                WHERE username = %s
            """
            cursor.execute(update_query, (formatted_end_date, username))
            conn.commit()

            if cursor.rowcount == 0:
                # No rows were updated, likely because the username does not exist
                return jsonify({'success': False, 'message': 'Username does not exist or is incorrect.'}), 404

            return jsonify({'success': True, 'message': 'Subscription will be cancelled at the end of the billing period.', 'end_date': formatted_end_date}), 200
        else:
            return jsonify({'success': False, 'message': 'No active subscription found.'}), 404
    except Exception as e:
        # Log the error here if needed
        print(f"Error: {str(e)}")  # Example logging
        tb = traceback.format_exc()  # This gets the full traceback as a string
        # Return the error message and a suitable HTTP status code
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": tb
            })# Be cautious about returning raw tracebacks in production environments
    finally:
        cursor.close()
        conn.close()


@app.route('/latest-task-status', methods=['GET'])
def get_latest_task_status():
    user_id = request.args.get('user_id')

    if not user_id:
        return jsonify({"error": "Missing 'user_id' parameter."}), 400

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # Modify the SQL query to filter by user_id
        cursor.execute(
            "SELECT task_id, status, result FROM task_queue WHERE user_id = %s ORDER BY task_id DESC LIMIT 1",
            (user_id,)
        )

        task = cursor.fetchone()
        cursor.close()
        conn.close()

        if task:
            return jsonify(task), 200
        else:
            return jsonify({"error": "No tasks found for the specified user."}), 404
    except mysql.connector.Error as err:
        return jsonify({"error": str(err)}), 500

@app.route('/get-user-details')
def get_user_details():
    if 'user' in session:
        username = session['user']
        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()
            cursor.execute("SELECT start_date, has_paid, subscription_end_date FROM users WHERE username = %s", (username,))
            user_data = cursor.fetchone()
            cursor.close()
            conn.close()

            if user_data:
                start_date, has_paid, subscription_end_date = user_data
                trial_end_date = start_date + timedelta(days=7)
                # Format subscription_end_date only if it is not None
                subscription_end_date_formatted = subscription_end_date.strftime('%Y-%m-%dT%H:%M:%SZ') if subscription_end_date else None
                return jsonify({
                    'success': True,
                    'trial_end_date': trial_end_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'subscription_end_date_formatted': subscription_end_date_formatted,  # Use the formatted variable directly
                    'has_paid': has_paid
                })
            else:
                return jsonify({'success': False, 'message': 'User not found'}), 404
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500
    else:
        return jsonify({'success': False, 'message': 'Not logged in'}), 403


@app.route('/reactivate-subscription', methods=['POST'])
def reactivate_subscription():
    data = request.get_json()
    username = data.get('username')  # Ensure this matches the email used in Stripe

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(buffered=True)

        # Retrieve the Stripe customer ID(s) associated with the email
        customers = stripe.Customer.list(email=username)
        if not customers.data:
            return jsonify({'success': False, 'message': 'No Stripe customer found with this email.'}), 404

        # Assuming you want to cancel subscriptions for the first customer object
        customer_id = customers.data[0].id

        # Fetch active subscriptions for the customer
        subscriptions = stripe.Subscription.list(customer=customer_id, status='active')

        if subscriptions.data:
            subscription = subscriptions.data[0]

            # Mark the subscription to cancel at the end of the billing period
            stripe.Subscription.modify(subscription.id, cancel_at_period_end=False)

            update_query = """
                UPDATE users
                SET subscription_end_date = %s, subscription_status = 'active'
                WHERE username = %s
            """
            # Pass None to effectively set the column to NULL in MySQL
            cursor.execute(update_query, (None, username))
            conn.commit()

            if cursor.rowcount == 0:
                # No rows were updated, likely because the username does not exist
                return jsonify({'success': False, 'message': 'Username does not exist or is incorrect.'}), 404

            return jsonify({'success': True, 'message': 'Subscription is reactivated'}), 200
        else:
            return jsonify({'success': False, 'message': 'No active subscription found.'}), 404
    except Exception as e:
        # Log the error here if needed
        print(f"Error: {str(e)}")  # Example logging
        tb = traceback.format_exc()  # This gets the full traceback as a string
        # Return the error message and a suitable HTTP status code
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": tb
            })# Be cautious about returning raw tracebacks in production environments
    finally:
        cursor.close()
        conn.close()

@app.route('/run-script', methods=['POST'])
def start_script():
    data = request.get_json()
    action = data.get('action')
    choice = data.get('choice', '5')  # Default to '5' if not provided
    user_id = data.get('user_id')
    db_name = data.get('db_name')
    number_of_days = data.get('num_days', None)  # Get the number of days from the request
    user_folder = os.path.join(app.config['UPLOAD_FOLDER_BASE'], user_id)

    if not user_id or not db_name:
        return jsonify({"error": "Missing 'user_id' or 'db_name' in request."}), 400

    if action != 'ensemble' and action != 'ensemble_prediction':
        db_path = os.path.join(user_folder, f"{db_name}.db")
        process_data_helper.create_tables(db_path)
    else:
        db_names_input = data.get('db_name')  # This can be a string or a comma-separated list of strings
        # Assuming db_names_input for ensemble action is a comma-separated string of db names
        db_names = db_names_input.split(',')  # Split the string into a list based on comma
        # Remove any leading/trailing whitespace from each db name
        db_names = [name.strip() for name in db_names]
        db_paths = [os.path.join(user_folder, f"{name}.db") for name in db_names]

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        if action == 'train':
            mse_error, movement_accuracy = run_multiple_experiments(choice, action, db_path, number_of_days)

            # Convert mse_error to a Python float
            mse_error_jsonifiable = float(mse_error)

            return jsonify({"mse_error": mse_error_jsonifiable,
                            "accuracy": movement_accuracy}), 200
        elif action == 'evaluate':
            mse_error, movement_accuracy_percent = run_multiple_experiments(choice, action, db_path, number_of_days)

            # Convert mse_error to a Python float
            mse_error_jsonifiable = float(mse_error)

            return jsonify({"mse_error": mse_error_jsonifiable,
                            "accuracy": movement_accuracy_percent}), 200
        elif action == 'predict-next':
            last_day_predicted_sales, next_day_predicted_sales, predicted_movement = run_multiple_experiments(choice, action, db_path, number_of_days)
            return jsonify({"last_day_predicted_sales": last_day_predicted_sales,
                            "next_day_predicted_sales": next_day_predicted_sales,
                            "predicted_movement": predicted_movement}), 200
        elif action == 'prophet':
            mae, rmse, mape, movement_accuracy = run_multiple_experiments(choice, action, db_path, number_of_days)
            return jsonify({"mae": mae,
                            "rmse": rmse,
                            "mape": mape,
                            "movement_accuracy": movement_accuracy}), 200
        elif action == 'prophet-evaluate':
            mae, rmse, mape, movement_accuracy = run_multiple_experiments(choice, action, db_path, number_of_days)
            return jsonify({"mae": mae,
                            "rmse": rmse,
                            "mape": mape,
                            "movement_accuracy": movement_accuracy}), 200
        elif action == 'xgboost':
            rmse, movement_accuracy = run_multiple_experiments(choice, action, db_path, number_of_days)
            return jsonify({"rmse": rmse,
            "movement_accuracy": movement_accuracy}), 200
        elif action == 'xgboost-evaluate':
            rmse, movement_accuracy = run_multiple_experiments(choice, action, db_path, number_of_days)
            return jsonify({"rmse": rmse,
            "movement_accuracy": movement_accuracy}), 200
        elif action == 'xgboost-prediction':
            predictions_dict = run_multiple_experiments(choice, action, db_path, number_of_days)
            return jsonify({"message": 'completed successfully',
                            "Predictions": predictions_dict}), 200
        elif action == 'prophet-predict':
            predictions_dict, dataset_ids = run_multiple_experiments(choice, action, db_path, number_of_days)
            return jsonify({"message": 'completed successfully',
                            "Predictions": predictions_dict,
                            "dataset_ids": dataset_ids}), 200
        elif action == 'ensemble':
            xgboost_dbs, prophet_dbs, lstm_dbs, db_name_verified, xgboost_predictions, prophet_predictions, LSTM_predictions, rmse, movement_accuracy, dataset_ids = run_multiple_experiments(choice, action, db_paths, number_of_days)
            return jsonify({"xgboost_dbs": xgboost_dbs,
                            "prophet_dbs": prophet_dbs,
                            "lstm_dbs": lstm_dbs,
                            "db_paths": db_paths,
                            "db_name_verified": db_name_verified,
                            "xgboost_predictions": xgboost_predictions,
                            "ensemble_predictions": prophet_predictions,
                            "LSTM_predictions": LSTM_predictions,
                            "rmse": rmse,
                            "movement_accuracy": movement_accuracy,
                            "dataset_ids": dataset_ids}), 200
        elif action == 'ensemble_prediction':
            number_of_days_int, xgboost_predictions, prophet_predictions, LSTM_predictions = run_multiple_experiments(choice, action, db_paths, number_of_days)
            return jsonify({"db_paths": db_paths,
                            "xgboost_predictions": xgboost_predictions,
                            "prophet_predictions": prophet_predictions,
                            "LSTM_predictions": LSTM_predictions,
                            "number_of_days_int": number_of_days_int}), 200
    except Exception as e:
        # Log the error here if needed
        print(f"Error: {str(e)}")  # Example logging
        tb = traceback.format_exc()  # This gets the full traceback as a string
        # Return the error message and a suitable HTTP status code
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": tb  # Be cautious about returning raw tracebacks in production environments
        }), 500

@app.route('/download-predictions', methods=['POST'])
def download_predictions():
    data = request.get_json()
    user_id = data.get('user_id')
    db_name = data.get('db_name')
    action = data.get('action')
    dataset_id = 'saved_predictions'

    if action =='ensemble_prediction':
        dataset_id = 'save_all_predictions'
        user_folder = os.path.join(app.config['UPLOAD_FOLDER_BASE'], user_id)
        db_names_input = data.get('db_name')  # This can be a string or a comma-separated list of strings
        # Assuming db_names_input for ensemble action is a comma-separated string of db names
        db_names = db_names_input.split(',')  # Split the string into a list based on comma
        # Remove any leading/trailing whitespace from each db name
        db_names = [name.strip() for name in db_names]
        db_paths = [os.path.join(user_folder, f"{name}.db") for name in db_names]
        # Select the first database path from the list
        first_db_path = db_paths[0] if db_paths else None
        # Constructing the path to the user's database
        user_folder = os.path.join(app.config['UPLOAD_FOLDER_BASE'], user_id)

        conn = process_data_helper.connect_to_db(first_db_path)

        # Fetching the plot data
        predictions_data = process_data_helper.retrieve_dataset_from_db(conn, dataset_id)

        if not predictions_data.empty:
            output = io.StringIO()
            predictions_data.to_csv(output, index=False)
            output.seek(0)
            return Response(
                output.getvalue(),
                mimetype='text/csv',
                headers={"Content-disposition":
                         "attachment; filename=predictions.csv"})
        else:
            return jsonify({"error": "Data not found"}), 404
    else:
        # Constructing the path to the user's database
        user_folder = os.path.join(app.config['UPLOAD_FOLDER_BASE'], user_id)
        db_path = os.path.join(user_folder, f"{db_name}.db")

        conn = process_data_helper.connect_to_db(db_path)

        # Fetching the plot data
        predictions_data = process_data_helper.retrieve_dataset_from_db(conn, dataset_id)

        if not predictions_data.empty:
            output = io.StringIO()
            predictions_data.to_csv(output, index=False)
            output.seek(0)
            return Response(
                output.getvalue(),
                mimetype='text/csv',
                headers={"Content-disposition":
                         "attachment; filename=predictions.csv"})
        else:
            return jsonify({"error": "Data not found"}), 404

@app.route('/download-plot', methods=['POST'])
def download_plot():
    data = request.get_json()
    user_id = data.get('user_id')
    db_name = data.get('db_name')
    action = data.get('action')
    if action == 'prophet' or action == 'prophet-evaluate':
        plot_name = 'prophet_plot'
    elif action == 'xgboost' or action == 'xgboost-evaluate':
        plot_name = 'xgboost_plot'
    elif action == 'xgboost-prediction':
        plot_name = 'xgboost_plot-prediction'
    elif action == 'predict-next':
        plot_name = 'predict-next_plot'
    elif action == 'prophet-predict':
        plot_name = 'prophet-predict_plot'
    elif action == 'ensemble':
        plot_name = 'ensemble_evaluate_plot'
    elif action == 'ensemble_prediction':
        plot_name = 'ensemble_prediction_plot'
    elif action == 'train' or action == 'evaluate':
        plot_name = 'MLP_plot_predictions_vs_actuals'

    if action == 'ensemble' or action == 'ensemble_prediction':

        user_folder = os.path.join(app.config['UPLOAD_FOLDER_BASE'], user_id)
        db_names_input = data.get('db_name')  # This can be a string or a comma-separated list of strings
        # Assuming db_names_input for ensemble action is a comma-separated string of db names
        db_names = db_names_input.split(',')  # Split the string into a list based on comma
        # Remove any leading/trailing whitespace from each db name
        db_names = [name.strip() for name in db_names]
        db_paths = [os.path.join(user_folder, f"{name}.db") for name in db_names]
        # Select the first database path from the list
        first_db_path = db_paths[0] if db_paths else None
        # Constructing the path to the user's database
        user_folder = os.path.join(app.config['UPLOAD_FOLDER_BASE'], user_id)

        # Fetching the plot data
        plot_data = process_data_helper.fetch_plot_from_db(first_db_path, plot_name)
        if plot_data:
            # Ensuring we're working with binary data correctly
            return send_file(
                io.BytesIO(plot_data),
                attachment_filename=f"{plot_name}.png",
                mimetype='image/png'
            )
        else:
            return jsonify({"error": "Plot not found"}), 404

    else:
        # Constructing the path to the user's database
        user_folder = os.path.join(app.config['UPLOAD_FOLDER_BASE'], user_id)
        db_path = os.path.join(user_folder, f"{db_name}.db")

        # Fetching the plot data
        plot_data = process_data_helper.fetch_plot_from_db(db_path, plot_name)
        if plot_data:
            # Ensuring we're working with binary data correctly
            return send_file(
                io.BytesIO(plot_data),
                attachment_filename=f"{plot_name}.png",
                mimetype='image/png'
            )
        else:
            return jsonify({"error": "Plot not found"}), 404


@app.route('/logout')
def logout():
    # Here you would clear the session or any other logout logic you need
    session.pop('user_id', None)  # Adjust according to how you've stored the user info
    return redirect(url_for('home'))  # Redirect to the main page

@app.route('/download-feature-importance', methods=['POST'])
def download_feature_importance():
    data = request.get_json()
    user_id = data.get('user_id')
    db_name = data.get('db_name')
    action = data.get('action')
    plot_name = 'xgboost_feature_importance'

    # Constructing the path to the user's database
    user_folder = os.path.join(app.config['UPLOAD_FOLDER_BASE'], user_id)
    db_path = os.path.join(user_folder, f"{db_name}.db")

    # Fetching the plot data
    plot_data = process_data_helper.fetch_plot_from_db(db_path, plot_name)
    if plot_data:
        # Ensuring we're working with binary data correctly
        return send_file(
            io.BytesIO(plot_data),
            attachment_filename=f"{plot_name}.png",
            mimetype='image/png'
        )
    else:
        return jsonify({"error": "Plot not found"}), 404