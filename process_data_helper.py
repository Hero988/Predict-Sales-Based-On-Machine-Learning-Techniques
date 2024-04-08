import pandas as pd  # For handling datasets
import os

from datetime import datetime, timedelta
import requests

import numpy as np  # For numerical operations

import sqlite3

import pickle

import torch
import io

def process_feature_set(filename):
    try:
        data = pd.read_csv(filename, parse_dates=['Date'], dayfirst=True)
    except UnicodeDecodeError:
        print("Encountered an encoding error, trying with ISO-8859-1 encoding.")
        data = pd.read_csv(filename, encoding='ISO-8859-1', parse_dates=['Date'], dayfirst=True)

    # Convert 'IsHoliday' column to integers (1 for True, 0 for False)
    data['IsHoliday'] = data['IsHoliday'].astype(bool).astype(int)

    # Replace 'NA' with numpy's NaN to enable filling with zeros
    data.replace('NA', np.nan, inplace=True)

    # Fill NaN values with 0 for Markdowns
    markdown_columns = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    data[markdown_columns] = data[markdown_columns].fillna(0)

    # Convert Markdown columns to numeric type in case they were read as objects
    data[markdown_columns] = data[markdown_columns].apply(pd.to_numeric)

    # Base directory to save store-wise folders
    base_path = 'feature_data_by_store'
    os.makedirs(base_path, exist_ok=True)

    for store in data['Store'].unique():
        store_data = data[data['Store'] == store]

        # Directory for the current store
        store_dir = os.path.join(base_path, f'store_{store}')
        os.makedirs(store_dir, exist_ok=True)

        # Save the store's data to a CSV file within its directory
        store_file_path = os.path.join(store_dir, f'store_{store}_features.csv')
        store_data.to_csv(store_file_path, index=False)

        print(f"Feature data for store {store} saved to '{store_file_path}'")

    return base_path

def calculate_daily_sales(filename):
    # Step 1: Read the original CSV file
    dataframe = pd.read_csv(filename)

    # Convert InvoiceDate to datetime to extract the date part only
    dataframe['InvoiceDate'] = pd.to_datetime(dataframe['InvoiceDate']).dt.date

    # Calculate the total sale for each line
    dataframe['TotalSale'] = dataframe['Quantity'] * dataframe['UnitPrice']

    # Group by InvoiceDate and sum up TotalSale for each date
    daily_sales = dataframe.groupby('InvoiceDate')['TotalSale'].sum().reset_index()

    # Rename columns for clarity
    daily_sales.columns = ['Date', 'sales_data_current']

    # Convert 'Date' to datetime format
    daily_sales['Date'] = pd.to_datetime(daily_sales['Date'], format='%d/%m/%Y')

    # Sort data by store and date to ensure chronological order
    daily_sales.sort_values(by=['Date'], inplace=True)

    daily_sales.rename(columns={'Date': 'date'}, inplace=True)

    daily_sales['sales_data_previous'] = daily_sales['sales_data_current'].shift(1)
    daily_sales['sales_data_percentage_change'] = daily_sales['sales_data_current'].pct_change() * 100
    daily_sales['previous_sales_data_percentage_change'] = daily_sales['sales_data_percentage_change'].shift(1)
    daily_sales['Actual Movement'] = np.where(daily_sales['sales_data_current'] > daily_sales['sales_data_previous'], 1,
                                                np.where(daily_sales['sales_data_current'] < daily_sales['sales_data_previous'], -1, 0))
    daily_sales['Previous Movement'] = daily_sales['Actual Movement'].shift(1).fillna(0)

    # Fill NaN values
    columns_to_fill = ['sales_data_previous', 'sales_data_percentage_change', 'previous_sales_data_percentage_change']
    daily_sales[columns_to_fill] = daily_sales[columns_to_fill].fillna(0)

    # Save the resulting DataFrame to a new CSV file named 'sales_data.csv'
    daily_sales.to_csv('sales_data_new.csv', index=False)

    print("Daily sales data has been successfully saved to 'sales_data.csv'.")

    return daily_sales


def process_sales_data_by_store(filename):
    try:
        # Load the data
        data = pd.read_csv(filename)
    except UnicodeDecodeError:
        print("Encountered an encoding error, trying with ISO-8859-1 encoding.")
        data = pd.read_csv(filename, encoding='ISO-8859-1')

    # Convert 'Date' to datetime format
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

    # Rename columns for clarity
    data.rename(columns={'Weekly_Sales': 'sales_data_current', 'Date': 'date'}, inplace=True)

    # Sort data by store and date to ensure chronological order
    data.sort_values(by=['Store', 'date'], inplace=True)

    # Base path for saving data
    base_path = 'sales_data_by_store'
    os.makedirs(base_path, exist_ok=True)

    # Get unique stores
    unique_stores = data['Store'].unique()

    for store in unique_stores:
        store_data = data[data['Store'] == store]

        # Process sales data for the store
        store_data['sales_data_previous'] = store_data['sales_data_current'].shift(1)
        store_data['sales_data_percentage_change'] = store_data['sales_data_current'].pct_change() * 100
        store_data['previous_sales_data_percentage_change'] = store_data['sales_data_percentage_change'].shift(1)
        store_data['Actual Movement'] = np.where(store_data['sales_data_current'] > store_data['sales_data_previous'], 1,
                                                 np.where(store_data['sales_data_current'] < store_data['sales_data_previous'], -1, 0))
        store_data['Previous Movement'] = store_data['Actual Movement'].shift(1).fillna(0)

        # Fill NaN values
        columns_to_fill = ['sales_data_previous', 'sales_data_percentage_change', 'previous_sales_data_percentage_change']
        store_data[columns_to_fill] = store_data[columns_to_fill].fillna(0)

        # Directory for the current store
        store_dir = os.path.join(base_path, f'store_{store}')
        os.makedirs(store_dir, exist_ok=True)

        # Save the store's data to a CSV file within its directory
        store_file_path = os.path.join(store_dir, f'store_{store}_combined_sales_data.csv')
        store_data.to_csv(store_file_path, index=False)

        print(f"Combined data for store {store} saved to '{store_file_path}'")

    return base_path

# Function to get latitude and longitude
def get_lat_long(city_name):
    # Load the dataset (adjust the path to where you've saved your dataset) (https://simplemaps.com/data/world-cities)
    df = pd.read_csv('worldcities.csv')
    city_data = df[df['city_ascii'] == city_name].iloc[0]  # Assuming city names are unique for simplicity
    return city_data['lat'], city_data['lng']


def get_temperature_and_save_to_csv(city, start_date, end_date):
    filename_temp = f'{city}_{start_date}_to_{end_date}.csv'
    lat, long = get_lat_long(city)
    print(lat, long)
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    all_data = []

    current_date = start_date
    while current_date <= end_date:
        url = "https://ai-weather-by-meteosource.p.rapidapi.com/time_machine"
        querystring = {"date": current_date.strftime("%Y-%m-%d"), "lat": f"{lat}", "lon": f"{long}", "units":"auto"}

        headers = {
            "X-RapidAPI-Key": "5d510b0614msh82ce58c3f1583d7p1dfc19jsnbd7d5f6b03b1",
            "X-RapidAPI-Host": "ai-weather-by-meteosource.p.rapidapi.com"
        }

        response = requests.get(url, headers=headers, params=querystring)
        response_data = response.json()  # Correctly assigning the response data

        # Check if 'data' key exists in the response_data
        if 'data' in response_data:
            daily_data = response_data['data']  # Ensure daily_data refers to the actual data list

            # Initialize accumulators for each attribute
            attributes = ['temperature', 'feels_like', 'wind_chill', 'dew_point',
                          'cloud_cover', 'pressure', 'ozone', 'humidity']
            accumulators = {attr: [] for attr in attributes}
            accumulators['speed'] = []
            accumulators['gusts'] = []
            accumulators['angle'] = []
            accumulators['total'] = []

            # Accumulate data from each hour
            for hour in daily_data:
                for attr in attributes:
                    accumulators[attr].append(hour.get(attr, 0))
                accumulators['speed'].append(hour['wind'].get('speed', 0))
                accumulators['gusts'].append(hour['wind'].get('gusts', 0))
                accumulators['angle'].append(hour['wind'].get('angle', 0))
                accumulators['total'].append(hour['precipitation'].get('total', 0))

            # Calculate averages and prepare the day's summary
            day_summary = {attr: sum(accumulators[attr]) / len(accumulators[attr]) for attr in accumulators}
            day_summary['Date'] = current_date.strftime("%Y-%m-%d")

            all_data.append(day_summary)
        else:
            # Rate Limit exceeded
            print(f"No 'data' key found for date: {current_date.strftime('%Y-%m-%d')} as Rate Limit exceeded")
            break

        current_date += timedelta(days=1)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(all_data)
    df.to_csv(filename_temp, index=False)
    print(f"Data saved to {filename_temp}")

    return filename_temp

def get_user_date_input(prompt):
    """
    Prompts the user for a date input and returns the date in 'YYYY-MM-DD' format.
    Continues to prompt until a valid date is entered.
    """
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

def extract_and_save_daily_sales_Brazilian_ECommerce(orders_path, order_items_path, order_payments_path):
    # Load datasets
    orders_df = pd.read_csv(orders_path)
    order_items_df = pd.read_csv(order_items_path)
    order_payments_df = pd.read_csv(order_payments_path)

    # Merge datasets
    merged_df = pd.merge(order_items_df, orders_df, on='order_id', how='inner')
    merged_df = pd.merge(merged_df, order_payments_df, on='order_id', how='inner')

    # Convert 'order_purchase_timestamp' to datetime and extract the date part
    merged_df['order_purchase_timestamp'] = pd.to_datetime(merged_df['order_purchase_timestamp'])
    merged_df['order_date'] = merged_df['order_purchase_timestamp'].dt.date

    # Calculate daily sales
    daily_sales = merged_df.groupby('order_date')['payment_value'].sum().reset_index()

    # Rename columns for clarity
    daily_sales.rename(columns={'payment_value': 'sales_data_current', 'order_date': 'date'}, inplace=True)

    # Convert 'Date' to datetime format
    daily_sales['date'] = pd.to_datetime(daily_sales['date'], format='%Y/%m/%d')

    # Process sales data for the store
    daily_sales['sales_data_previous'] = daily_sales['sales_data_current'].shift(1)
    daily_sales['sales_data_percentage_change'] = daily_sales['sales_data_current'].pct_change() * 100
    daily_sales['previous_sales_data_percentage_change'] = daily_sales['sales_data_percentage_change'].shift(1)
    daily_sales['Actual Movement'] = np.where(daily_sales['sales_data_current'] > daily_sales['sales_data_previous'], 1,
                                                np.where(daily_sales['sales_data_current'] < daily_sales['sales_data_previous'], -1, 0))
    daily_sales['Previous Movement'] = daily_sales['Actual Movement'].shift(1).fillna(0)

    # Fill NaN values
    columns_to_fill = ['sales_data_previous', 'sales_data_percentage_change', 'previous_sales_data_percentage_change']
    daily_sales[columns_to_fill] = daily_sales[columns_to_fill].fillna(0)

    output_folder=r'Brazilian_E-Commerce_Sales_Choice_4_Folder'
    output_file='sales_data.csv'

    # Ensure the output folder exists, create if it doesn't
    os.makedirs(output_folder, exist_ok=True)

    # Full path to the output file
    output_path = os.path.join(output_folder, output_file)

    # Save to CSV in the specified folder
    daily_sales.to_csv(output_path, index=False)
    print(f"Daily sales data saved to {output_path}.")

def connect_to_db(db_name):
    """Establish a connection to the SQLite database."""
    return sqlite3.connect(db_name)

def save_dataset_to_db(conn, dataset_id, dataset_name, df):
    """Save or update a DataFrame as a serialized dataset in the database."""
    serialized_data = pickle.dumps(df)
    cursor = conn.cursor()

    # Use UPSERT operation (INSERT ON CONFLICT UPDATE)
    cursor.execute("""
        INSERT INTO datasets (dataset_id, dataset_name, data)
        VALUES (?, ?, ?)
        ON CONFLICT(dataset_id)
        DO UPDATE SET
            dataset_name = excluded.dataset_name,
            data = excluded.data;
    """, (dataset_id, dataset_name, serialized_data))

    conn.commit()

def get_all_dataset_ids(db_name):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Execute a query to select all dataset IDs
    cursor.execute("SELECT dataset_id FROM datasets")

    # Fetch all results
    dataset_ids = cursor.fetchall()

    # Close the connection
    conn.close()

    # Extract dataset IDs from the tuples returned by fetchall()
    dataset_ids = [id[0] for id in dataset_ids]

    return dataset_ids

def retrieve_dataset_from_db(conn, dataset_id):
    """Retrieve a dataset by its ID."""
    cursor = conn.cursor()
    cursor.execute("SELECT data FROM datasets WHERE dataset_id = ?", (dataset_id,))
    row = cursor.fetchone()
    if row:
        return pickle.loads(row[0])
    return None

def save_log_to_db(conn, log_id, experiment_id, log_df):
    """Save a log DataFrame to the database."""
    serialized_log = log_df.to_json()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO log_files (log_id, experiment_id, log_data) VALUES (?, ?, ?)",
                   (log_id, experiment_id, serialized_log))
    conn.commit()

def fetch_log_from_db(conn, log_id):
    cursor = conn.cursor()
    cursor.execute("SELECT log_data FROM log_files WHERE log_id = ?", (log_id,))
    row = cursor.fetchone()
    if row:
        log_df = pd.read_json(row[0])
        return log_df
    else:
        return None

def save_predictions_to_db(conn, result_id, experiment_id, predictions_df):
    """Save predictions DataFrame to the database."""
    serialized_predictions = predictions_df.to_json()
    cursor = conn.cursor()
    cursor.execute("REPLACE INTO prediction_results (result_id, experiment_id, predictions_data) VALUES (?, ?, ?)",
                   (result_id, experiment_id, serialized_predictions))
    conn.commit()

def fetch_predictions_from_db(conn, result_id):
    cursor = conn.cursor()
    cursor.execute("SELECT predictions_data FROM prediction_results WHERE result_id = ?", (result_id,))
    row = cursor.fetchone()
    if row:
        predictions_df = pd.read_json(row[0])
        return predictions_df
    else:
        return None

def save_model_to_db(conn, model_name, model, experiment_id):
    """
    Save a PyTorch model's state to the database.

    Parameters:
    - conn: The database connection object.
    - model_name: A custom name for the model.
    - model: The PyTorch model whose state dictionary you want to save.
    - experiment_id: A unique identifier for the experiment.
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)  # Go to the beginning of the buffer to ensure all data is read
    serialized_model_state = buffer.getvalue()

    # Prepare the SQL command and data
    sql_command = "INSERT INTO models (model_name, model_state, experiment_id) VALUES (?, ?, ?)"
    data = (model_name, serialized_model_state, experiment_id)

    # Execute the command
    cursor = conn.cursor()
    cursor.execute(sql_command, data)
    conn.commit()

def save_prophet_model_to_db(conn, model_name, serialized_model, experiment_id=None):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO models (model_name, model_state, experiment_id) VALUES (?, ?, ?)",
                   (model_name, serialized_model, experiment_id))
    conn.commit()

def retrieve_model_from_db_prophet_xgboost(conn, model_name):
    cursor = conn.cursor()
    # Assuming your table is named 'models' and has a 'model_name' and 'model_state' columns
    cursor.execute("SELECT model_state FROM models WHERE model_name = ?", (model_name,))
    row = cursor.fetchone()

    if row:
        serialized_model = row[0]
        model = pickle.loads(serialized_model)
        return model
    else:
        print("Model not found in the database.")
        return None

# Example usage:
# Assuming `conn` is your database connection object
# model_name = "YourModelNameHere"
# prophet_model = retrieve_model_from_db(conn, model_name)
# if prophet_model:
#     # You can now use the prophet_model for predictions
#     future = prophet_model.make_future_dataframe(periods=365)
#     forecast = prophet_model.predict(future)

def save_scaler_to_db(conn, scaler_name, scaler):
    # Serialize the scaler object
    serialized_scaler = pickle.dumps(scaler)

    # Execute the insert or update command
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO scalers (scaler_name, scaler_object)
        VALUES (?, ?)
        ON CONFLICT(scaler_name) DO UPDATE SET scaler_object = excluded.scaler_object;
    """, (scaler_name, serialized_scaler))
    conn.commit()

def load_scaler_from_db(conn, scaler_name):
    cursor = conn.cursor()
    cursor.execute("SELECT scaler_object FROM scalers WHERE scaler_name = ?", (scaler_name,))
    row = cursor.fetchone()
    if row:
        # Deserialize the scaler object
        scaler = pickle.loads(row[0])
        return scaler
    return None

def retrieve_model_from_db(conn, model_name=None, experiment_id=None):
    """
    Retrieve a PyTorch model's state dictionary from the database.

    Parameters:
    - conn: The database connection object.
    - model_name: The name of the model to retrieve.
    - experiment_id: The unique identifier of the experiment. This or model_name must be provided.

    Returns:
    - The loaded state dictionary of the model if found, None otherwise.
    """
    cursor = conn.cursor()

    # Query based on provided identifier
    if model_name:
        cursor.execute("SELECT model_state FROM models WHERE model_name = ?", (model_name,))
    elif experiment_id:
        cursor.execute("SELECT model_state FROM models WHERE experiment_id = ?", (experiment_id,))
    else:
        print("Error: Please provide either a model_name or experiment_id.")
        return None

    row = cursor.fetchone()
    if row:
        # Load the model state from the byte stream
        buffer = io.BytesIO(row[0])
        model_state = torch.load(buffer)
        return model_state
    else:
        return None


def create_tables(db_name):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Create the 'experiments' table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS experiments (
        experiment_id TEXT PRIMARY KEY,
        choice TEXT,
        config TEXT, -- JSON serialized configuration
        metrics TEXT, -- JSON serialized metrics
        model_state BLOB, -- Serialized model state dictionary
        dataset_id TEXT -- Link to the dataset used for the experiment
    );
    """)

    # Create the 'datasets' table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS datasets (
        dataset_id TEXT PRIMARY KEY,
        dataset_name TEXT,
        data BLOB -- Serialized dataset
    );
    """)

    # Create the 'scalers' table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS scalers (
        scaler_name TEXT PRIMARY KEY,
        scaler_object BLOB NOT NULL
    );
    """)

    # Create the 'models' table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS models (
        model_id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        model_state BLOB NOT NULL, -- The serialized model state
        experiment_id TEXT NOT NULL,
        FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
    );
    """)

    # Add table for log files
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS log_files (
        log_id TEXT PRIMARY KEY,
        experiment_id TEXT,
        log_data TEXT, -- Serialized log data as text
        FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
    );
    """)

    # Add table for prediction results
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prediction_results (
        result_id TEXT PRIMARY KEY,
        experiment_id TEXT,
        predictions_data TEXT, -- Serialized prediction results data as text
        FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
    );
    """)

    # Create the 'plots' table for storing serialized plot data
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS plots (
        plot_id INTEGER PRIMARY KEY AUTOINCREMENT,
        plot_name TEXT UNIQUE NOT NULL,  -- Ensure plot_name is unique to enable upsert
        plot_data BLOB NOT NULL  -- The serialized plot data
    );
    """)

    # Commit the changes and close the connection
    conn.commit()
    conn.close()


def delete_model_from_db(conn, model_name):
    try:
        # Create a cursor object using the connection
        cursor = conn.cursor()

        # SQL statement to delete a model based on its name
        sql = "DELETE FROM models WHERE model_name = ?"

        # Execute the SQL command, using model_name to identify the model to delete
        cursor.execute(sql, (model_name,))

        # Commit the changes to the database
        conn.commit()
    except sqlite3.Error as e:
        # In case of any error, print the error message
        print(f"Error deleting model from database: {e}")

def save_plot_to_db(db_name, plot_name, plot_data):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    # Assuming `plot_data` is a binary object representing the image
    cursor.execute("""
        INSERT INTO plots (plot_name, plot_data)
        VALUES (?, ?)
        ON CONFLICT(plot_name) DO UPDATE SET plot_data = excluded.plot_data;
    """, (plot_name, plot_data))
    conn.commit()
    conn.close()

def fetch_plot_from_db(db_name, plot_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("SELECT plot_data FROM plots WHERE plot_name = ?", (plot_name,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return row[0]
    return None

def read_user_csv_and_process_data(db_name, filepath, action):
    if not filepath:
        print("No file was selected.")
        return False

    try:
        df = pd.read_csv(filepath)

        if 'sales_data_current' not in df.columns:
            print("The CSV must contain a 'sales_data_current' column.")
            return False

        if 'date' not in df.columns:
            print("No 'date' column found. Assuming the first column is the date.")
            date_col = df.columns[0]
        else:
            date_col = 'date'

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        if df[date_col].isnull().any():
            print("Failed to convert some date values to datetime format. Check date format.")
            return False

        # Now, set the index to the datetime column
        df.set_index(date_col, inplace=True)

        # Since index is now a DatetimeIndex, we can sort by it directly
        df.sort_index(inplace=True)

        lags = 7

        target_name = 'sales_data_current'

        for lag in range(1, lags + 1):
            df[f'{target_name}_lag_{lag}'] = df[target_name].shift(lag)

        # Reset the index to make 'date' a column, while preserving the index as a separate column
        df.reset_index(inplace=True)

        # Generate the additional columns based on the datetime index
        df['sales_data_previous'] = df['sales_data_current'].shift(1).fillna(0)
        df['sales_data_percentage_change'] = df['sales_data_current'].pct_change().fillna(0) * 100
        df['previous_sales_data_percentage_change'] = df['sales_data_percentage_change'].shift(1).fillna(0)
        df['Actual Movement'] = np.where(df['sales_data_current'] > df['sales_data_previous'], 1, np.where(df['sales_data_current'] < df['sales_data_previous'], -1, 0))
        df['Previous Movement'] = df['Actual Movement'].shift(1).fillna(0)

        # These operations are valid as the index is a datetime
        df['dayofweek'] = pd.to_datetime(df[date_col]).dt.dayofweek
        df['quarter'] = pd.to_datetime(df[date_col]).dt.quarter
        df['month'] = pd.to_datetime(df[date_col]).dt.month
        df['year'] = pd.to_datetime(df[date_col]).dt.year
        df['dayofyear'] = pd.to_datetime(df[date_col]).dt.dayofyear

        # Replace 'inf' values with 'NaN'
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop rows with 'NaN' in any column
        df.fillna(0, inplace=True)

        # Optionally, you can reset the index if you want a continuous index after dropping rows
        df.reset_index(drop=True, inplace=True)

        # Example of saving the processed DataFrame to a database
        conn = connect_to_db(db_name)
        if action == 'train':
            save_dataset_to_db(conn, 'processed_data', 'User Provided Dataset 1', df)
        elif action == 'predict-next':
            save_dataset_to_db(conn, 'processed_data_next', 'User Provided Dataset 2', df)
        elif action == 'prophet':
            save_dataset_to_db(conn, 'processed_data_for_prophet', 'User Provided Dataset 3', df)
        elif action == 'xgboost':
            save_dataset_to_db(conn, 'processed_data_for_xgboost', 'User Provided Dataset 4', df)
        elif action == 'ensemble':
            save_dataset_to_db(conn, 'processed_data_for_ensemble', 'User Provided Dataset 8', df)
        print("CSV loaded, processed, and saved to database successfully.")
        return True

    except pd.errors.EmptyDataError:
        print("The CSV file is empty. Please provide a valid CSV file.")
        return False
    except pd.errors.ParserError:
        print("Error parsing CSV. Please ensure the CSV file is formatted correctly.")
        return False
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        return False