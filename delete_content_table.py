import mysql.connector

def fetch_all_tasks():
    # Assuming you've set up your MySQL connection details
    db_config = {
        'user': 'Hero98',
        'password': 'Ameer1fadi@3',
        'host': 'Hero98.mysql.pythonanywhere-services.com',
        'database': 'Hero98$PredictPrice',
        'raise_on_warnings': True
    }
    try:
        # Establish a connection to the database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)  # Use dictionary=True to get results as dictionaries

        # Execute the query to fetch all data from the task_queue table
        cursor.execute("SELECT * FROM task_queue")

        # Fetch all rows from the result of the query execution
        tasks = cursor.fetchall()

        # Check if we got any results
        if tasks:
            for task in tasks:
                print(task)  # Print each task, which is a dictionary
        else:
            print("No tasks found in the task_queue table.")

    except mysql.connector.Error as err:
        # Handle potential errors that may occur during the process
        print(f"Error: {err}")
    finally:
        # Ensure the cursor and connection are closed to free resources
        if conn.is_connected():
            cursor.close()
            conn.close()
            print("MySQL connection is closed")

def delete_all():

    # Assuming you've set up your MySQL connection details
    db_config = {
        'user': 'Hero98',
        'password': 'Ameer1fadi@3',
        'host': 'Hero98.mysql.pythonanywhere-services.com',
        'database': 'Hero98$PredictPrice',
        'raise_on_warnings': True
    }

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Execute the SQL statement to delete all rows from the task_queue table
        cursor.execute("DELETE FROM task_queue")

        # Commit the changes
        conn.commit()

        print("All records have been successfully deleted from the task_queue table.")
    except mysql.connector.Error as err:
        print(f"Failed to delete records from table: {err}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()
            print("MySQL connection is closed")

# Call the function to fetch and print all tasks
fetch_all_tasks()