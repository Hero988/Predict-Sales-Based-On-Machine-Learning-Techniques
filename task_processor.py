import mysql.connector
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error
import io
import process_data_helper
import os
import traceback
"""
Yes, that's correct. Your LSTM model is designed to predict the next target value based on a sequence of feature values from previous time steps.
The model takes in sequences of feature data, processes them through a bidirectional LSTM layer (allowing it to learn from both past and future context within each sequence),
and outputs a prediction for the target value at the next time step following the sequence.
This design is typical for time-series forecasting tasks, where the goal is to use historical data points to predict future outcomes.
"""

class SalesPredictionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(SalesPredictionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout_rate,
                            batch_first=True,
                            bidirectional=True)

        self.dropout = nn.Dropout(dropout_rate)

        # No change in hidden_size * 2 due to bidirectional LSTM, but removing sigmoid
        self.output_layer = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out)
        out = out[:, -1, :]  # Keep this to use the last sequence output
        out = self.output_layer(out)
        # No sigmoid application for regression
        return out

def prepare_data_for_LSTM_training(retrieved_dataset, split_ratio=0.8):
    # Ensure 'date' column is in datetime format
    retrieved_dataset['date'] = pd.to_datetime(retrieved_dataset['date'])

    # Make a copy of the dataset
    prepared_dataset = retrieved_dataset.copy()

    # Ensure the dataset is sorted by date
    prepared_dataset.sort_values('date', inplace=True)

    # Calculate the split index based on the provided split_ratio
    split_index = int(len(prepared_dataset) * split_ratio)

    # Retrieve the first 80% of the dataset for training
    train_set = prepared_dataset.iloc[:split_index]

    return train_set

def prepare_data_for_LSTM_split_training(feature_names, target_name, train_set, seq_length):
    # Ensure the 'date' column is set as the index and is in datetime format
    train_set['date'] = pd.to_datetime(train_set['date'])
    train_set.set_index('date', inplace=True)

    # Replace infinite values with NaN
    train_set.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaN values with the mean of each column
    train_set.fillna(train_set.mean(), inplace=True)

    # Extract features (X) and target (y) from the DataFrame using provided names
    X = train_set[feature_names].values
    y = train_set[target_name].values.reshape(-1, 1)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create sequences
    def create_sequences(X_scaled, y, seq_length):
        Xs, ys = [], []
        for i in range(len(X_scaled) - seq_length):
            Xs.append(X_scaled[i:(i + seq_length)])
            ys.append(y[i + seq_length])
        return np.array(Xs), np.array(ys)

    X_sequences, y_sequences = create_sequences(X_scaled, y, seq_length)

    # Split the sequences into training and validation sets
    val_size = int(len(X_sequences) * 0.25)
    X_train, X_val = X_sequences[:-val_size], X_sequences[-val_size:]
    y_train, y_val = y_sequences[:-val_size], y_sequences[-val_size:]

    # Create TensorDatasets for the training and validation sets
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).squeeze())
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).squeeze())

    return train_dataset, val_dataset

def calculate_validation_error(model, val_loader):
    model.eval()  # Set the model to evaluation mode
    actuals = []
    predictions = []

    with torch.no_grad():  # Disable gradient computation
        for inputs, targets in val_loader:  # Iterate over the data
            outputs = model(inputs)  # Forward pass
            predictions.extend(outputs.numpy().flatten())  # Store predictions
            actuals.extend(targets.numpy().flatten())  # Store actual values

    # Calculate Mean Squared Error
    val_error = mean_squared_error(actuals, predictions)
    # Calculate Root Mean Squared Error using np.sqrt
    rmse = np.sqrt(val_error)
    model.train()  # Set the model back to train mode
    return rmse

def train_model_no_epoch_set(model, criterion, optimizer, train_loader, val_loader, model_saved, db_name, choice):
    # Patience for early stopping
    patience = 20
    lowest_val_error = float('inf')  # Initialize the lowest validation error
    epochs_no_improve = 0  # Counter for epochs with no improvement
    epoch = 0  # Initialize epoch counter

    print("Training started. Elapsed time: 0s", end="", flush=True)
    start_time = time.time()

    log_entries = []  # Initialize an empty list for log entries

    while epochs_no_improve < patience:  # Loop until the early stopping condition is met
        model.train()  # Set the model to training mode
        for inputs, targets in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs.squeeze(), targets)  # Calculate loss for regression
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

        # Validate the model and calculate validation error
        val_error = calculate_validation_error(model, val_loader)

        # Check for improvement
        if val_error < lowest_val_error:
            print(f"Current RMSE {val_error} is less than the lowest RMSE {lowest_val_error} resetting early stopping counter")
            lowest_val_error = val_error  # Update lowest validation error
            epochs_no_improve = 0  # Reset counter
            status = "Lower validation error found, saving model"
            if model_saved is not None:
                torch.save(model.state_dict(), model_saved)  # Save the model
            elif model_saved is None:
                # Usage example, integrated with checking, deletion, and saving of a new model
                conn = process_data_helper.connect_to_db(db_name)  # Make sure you define or replace `connect_to_db` with your actual database connection function
                model_name = "MyCustomModelName"
                experiment_id = "Experiment123"

                # Check if the model already exists in the database
                existing_model = process_data_helper.retrieve_model_from_db(conn, model_name=model_name)
                if existing_model is not None:
                    # Delete the existing model from the database
                    process_data_helper.delete_model_from_db(conn, model_name)

                # Save the new model to the database
                process_data_helper.save_model_to_db(conn, model_name, model, experiment_id)

                conn.close()  # Always remember to close the database connection when done
        else:
            epochs_no_improve += 1  # Increment counter
            status = "No improvement."

        # Log the current epoch's results
        log_entries.append({
            'Epoch': epoch + 1,
            'Training Loss': f"{loss.item():.4f}",
            'Validation Error': f"{val_error:.4f}",
            'Status': status
        })

        # Update elapsed time display every epoch
        elapsed_time = int(time.time() - start_time)
        print(f"\rTraining in progress. Elapsed time: {elapsed_time}s", end="", flush=True)

        epoch += 1  # Increment epoch counter

    print(f"\rTraining Completed. Elapsed time: {elapsed_time}s", flush=True)

def train_model_set_epoch(model, criterion, optimizer, train_loader, val_loader, model_saved, db_name, choice):
    lowest_val_error = float('inf')  # Initialize the lowest validation error
    best_epoch = 0  # Track the epoch number where the lowest validation error was observed

    num_epochs = 10

    print("Training started.")
    start_time = time.time()

    log_entries = []  # Initialize an empty list for log entries

    for epoch in range(num_epochs):  # Iterate over the fixed number of epochs
        model.train()  # Set the model to training mode
        for inputs, targets in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs.squeeze(), targets)  # Calculate loss for regression
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

        # Validate the model and calculate validation error
        val_error = calculate_validation_error(model, val_loader)

        # Check for improvement
        if val_error < lowest_val_error:
            lowest_val_error = val_error  # Update lowest validation error
            best_epoch = epoch  # Update the epoch number of the best model
            status = "Lower validation error found, saving model"
            if model_saved is not None:
                torch.save(model.state_dict(), model_saved)  # Save the model
            elif model_saved is None:
                # Usage example, integrated with checking, deletion, and saving of a new model
                conn = process_data_helper.connect_to_db(db_name)  # Make sure you define or replace `connect_to_db` with your actual database connection function
                model_name = "MyCustomModelName"
                experiment_id = "Experiment123"

                # Check if the model already exists in the database
                existing_model = process_data_helper.retrieve_model_from_db(conn, model_name=model_name)
                if existing_model is not None:
                    # Delete the existing model from the database
                    process_data_helper.delete_model_from_db(conn, model_name)

                # Save the new model to the database
                process_data_helper.save_model_to_db(conn, model_name, model, experiment_id)

                conn.close()  # Always remember to close the database connection when done
        else:
            status = "No improvement."

        # Log the current epoch's results
        log_entries.append({
            'Epoch': epoch + 1,
            'Training Loss': f"{loss.item():.4f}",
            'Validation Error': f"{val_error:.4f}",
            'Status': status
        })

        # Update elapsed time display every epoch
        elapsed_time = int(time.time() - start_time)
        print(f"\rEpoch {epoch+1}/{num_epochs}. Elapsed time: {elapsed_time}s", end="", flush=True)

    print(f"\nTraining Completed. Elapsed time: {elapsed_time}s. Best model found at epoch {best_epoch+1}.", flush=True)

def prepare_evaluation_data_for_LSTM(retrieved_dataset, split_ratio=0.8):
    # Ensure 'date' column is in datetime format
    retrieved_dataset['date'] = pd.to_datetime(retrieved_dataset['date'])

    # Make a copy of the dataset
    prepared_dataset = retrieved_dataset.copy()

    # Ensure the dataset is sorted by date
    prepared_dataset.sort_values('date', inplace=True)

    # Calculate the split index based on the provided split_ratio
    split_index = int(len(prepared_dataset) * split_ratio)

    # Retrieve the last 20% of the dataset for evaluation
    evaluation_set = prepared_dataset.iloc[split_index:]

    return evaluation_set

def load_and_preprocess_data_evaluate(feature_names, target_name, seq_length, evaluation_set):
    # Make a copy to preserve the original DataFrame, now directly using the evaluation_set
    df_original = evaluation_set.copy()

    # Since evaluation_set is already filtered, we skip date filtering

    # Now set the date as index for processing
    df = df_original.set_index('date')

    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Replace NaN values with a specified value or a strategy
    df.fillna(df.mean(), inplace=True)

    # Extract features (X) and target (y) from the DataFrame
    X = df[feature_names].values
    y = df[target_name].values.reshape(-1, 1)

    # Initialize a scaler to standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create sequences from the scaled features and target
    def create_sequences(X_scaled, y, seq_length):
        Xs, ys = [], []
        for i in range(len(X_scaled) - seq_length):
            Xs.append(X_scaled[i:(i + seq_length)])
            ys.append(y[i + seq_length])
        return np.array(Xs), np.array(ys)

    # Apply sequence creation
    X_sequences, y_sequences = create_sequences(X_scaled, y, seq_length)

    # Prepare the tensor dataset
    eval_dataset = TensorDataset(torch.FloatTensor(X_sequences), torch.FloatTensor(y_sequences).squeeze(1))

    # Adjust df_original to match sequences
    # Since sequences start at 'seq_length', adjust df_original to align with sequences
    df_evaluate_aligned = df_original.iloc[seq_length:].reset_index(drop=False)

    return eval_dataset, df_evaluate_aligned

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

def save_predictions_plot(df, db_name, action):
    if action == 'evaluate_train':
        plot_name = 'LSTM_plot'
    elif action == 'xgboost':
        plot_name = 'xgboost_plot'
    # Buffer to save image
    buf = io.BytesIO()

    if action == 'xgboost':
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


def run_multiple_experiments(choice, action, db_name):
    target_name = 'sales_data_current'
    feature_names = ['dayofweek', 'quarter', 'month', 'year', 'dayofyear']
    model_saved = None

    conn = process_data_helper.connect_to_db(db_name)
    dataset_id = "processed_data"
    retrieved_dataset = process_data_helper.retrieve_dataset_from_db(conn, dataset_id)
    conn.close()

    # Batch size for training: The number of samples (data points) that the model is exposed to before the optimizer updates the model parameters.
    # For instance, a batch size of 32 means the model processes 32 samples before making a single update to its parameters.
    batch_size = 64

    # LSTM model parameters
    hidden_size = 128  # Number of features in the hidden state
    num_layers = 3  # Two stacked LSTM layers
    dropout_rate = 0.5  # 50% probability of dropping a unit
    input_size = len(feature_names)  # Number of input features

    # How many previous data can the LSTM see before making a prediction
    seq_length = 24

    # Initialize the LSTM model
    model = SalesPredictionLSTM(input_size, hidden_size, num_layers, dropout_rate)

    # measure the difference between the accurate and the predicted
    criterion = nn.MSELoss()
    # Utilizes the Adam optimization algorithm with a learning rate of 0.001 and L2 regularization (weight decay) set to 1e-5 to minimize overfitting by penalizing large weights.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    if action == 'train':
        # Split the data
        train_set = prepare_data_for_LSTM_training(retrieved_dataset)
        # load and process the data
        train_dataset, val_dataset = prepare_data_for_LSTM_split_training(feature_names, target_name, train_set, seq_length)
        # Create data loaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        # Train the model with a set epoch, train_model_no_epoch_set - this is the proper function but cannot run it as I need to figure out how to send code to the pythonanywhere codebase
        train_model_set_epoch(model, criterion, optimizer, train_loader, val_loader, model_saved, db_name, choice)
        action = 'evaluate_train'
    if action == 'evaluate_train':
        # Split Data
        evaluation_set = prepare_evaluation_data_for_LSTM(retrieved_dataset)
        # Get DataSets
        test_dataset, test_dataframe= load_and_preprocess_data_evaluate(feature_names, target_name, seq_length, evaluation_set)
        # Create data loader
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        # Evaluate the model on test data
        mse_error, movement_accuracy_percent, test_dataframe_1 = evaluate_model(model, test_loader, test_dataframe, choice, model_saved, action, db_name)

        # Ensure 'date' is in the correct format and is a column in both DataFrames
        evaluation_set['date'] = pd.to_datetime(retrieved_dataset['date'])
        test_dataframe_1['date'] = pd.to_datetime(test_dataframe_1['date'])

        # Merge evaluation_set with test_dataframe_1 on 'date' to align 'Predicted' and other columns
        merged_dataset = pd.merge(
            evaluation_set,
            test_dataframe_1[['date', 'Predicted', 'Previous Predicted', 'Predicted Movement']],
            on='date',
            how='left'
        )
        save_predictions_plot(merged_dataset, db_name, action)
        return mse_error, movement_accuracy_percent


db_config = {
    'user': 'Hero98',
    'password': 'Ameer1fadi@3',
    'host': 'Hero98.mysql.pythonanywhere-services.com',
    'database': 'Hero98$PredictPrice',
}

def process_task():
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM task_queue WHERE status = 'queued' ORDER BY created_at LIMIT 1")
    task = cursor.fetchone()
    if task:
        print(f"Found task {task['task_id']}. Starting task...")
        try:
            # Update task status to 'processing'
            cursor.execute("UPDATE task_queue SET status = 'processing' WHERE task_id = %s", (task['task_id'],))
            conn.commit()

            base_dir = '/home/Hero98/mysite_testing/user_data'

            # Construct db_path based on received 'user_id' and 'db_name'
            user_folder = os.path.join(os.path.join(base_dir), task['user_id'])

            # Process the task
            # Simulate or perform actual task processing here
            mse_error, movement_accuracy_percent = run_multiple_experiments(task['choice'], task['action'], task['db_name'])

            # Assuming mse_error and movement_accuracy_percent are already calculated
            results_array = [mse_error, movement_accuracy_percent]

            # Serialize the results_array to a JSON string
            results_json = json.dumps(results_array)

            # Update task status to 'completed' with results_json as the result
            cursor.execute(
                "UPDATE task_queue SET status = 'completed', result = %s WHERE task_id = %s",
                (results_json, task['task_id'])
            )
            print(f"Task {task['task_id']} completed.")
        except Exception as e:
            tb = traceback.format_exc()
            # Update task status to 'failed' on exception
            cursor.execute("UPDATE task_queue SET status = 'failed', result = %s WHERE task_id = %s", (tb, task['task_id']))
            print(f"Task {task['task_id']} failed.")
        finally:
            conn.commit()
    else:
        print("No new tasks. Checking again...")

if __name__ == "__main__":
    while True:
        process_task()
        time.sleep(10)  # Check for new tasks every 10 seconds