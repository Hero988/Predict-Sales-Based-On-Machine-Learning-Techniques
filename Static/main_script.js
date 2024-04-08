document.addEventListener("DOMContentLoaded", function() {
    var urlParams = new URLSearchParams(window.location.search);
    var actionParam = urlParams.get('action');

    function handleActionChange() {
        var action = document.getElementById('action').value;
        var userForm = document.getElementById('userForm');
        var uploadForm = document.getElementById('uploadForm');
        var trainingStatusDisplay = document.getElementById('trainingStatusDisplay');
        var statusMessage = document.getElementById('status');
        var downloadPlotBtn = document.getElementById('downloadPlotBtn');
        var downloadCsvBtn = document.getElementById('downloadCsvBtn');
        var downloadFeatureImportanceBtn = document.getElementById('downloadFeatureImportanceBtn');
        var dbNameInput = document.getElementById('db_name'); // Get the db_name input element
        var dbNameSuffixContainer = document.getElementById('dbNameSuffix'); // Get the suffix container

        var numDaysContainer = document.getElementById('numDaysContainer');

        numDaysContainer.style.display = (action === 'xgboost-prediction' || action === 'prophet-predict' || action === 'ensemble_prediction') ? 'block' : 'none';

        var action_message_Message = document.getElementById('action_message');

        // Initialize a flag to determine whether to show the db_name input
        let showDbNameInput = false;

        let showDbNameSuffix = false;


        // Logic to set db_name based on selected action
        var dbNameSuffix = ''; // Suffix to append to db_name
        switch(action) {
            case 'train':
                dbNameSuffix = ' - MP DB';
                showDbNameInput = false; // Show for these actions
                showDbNameSuffix = false; // Show the suffix for these actions
                break;
            case 'evaluate':
                showDbNameInput = false; // Show for these actions
                showDbNameSuffix = false; // Show the suffix for these actions
                break;
            case 'predict-next':
                showDbNameInput = false; // Show for these actions
                showDbNameSuffix = false; // Show the suffix for these actions
                break;
            case 'prophet':
                dbNameSuffix = ' - Prophet DB';
                showDbNameInput = false; // Show for these actions
                showDbNameSuffix = false; // Show the suffix for these actions
                break;
            case 'prophet-evaluate':
                showDbNameInput = false; // Show for these actions
                showDbNameSuffix = false; // Show the suffix for these actions
                break;
            case 'xgboost':
                dbNameSuffix = ' - XGBoost DB';
                showDbNameInput = false; // Show for these actions
                showDbNameSuffix = false; // Show the suffix for these actions
                break;
            case 'xgboost-evaluate':
                showDbNameInput = false; // Show for these actions
                showDbNameSuffix = false; // Show the suffix for these actions
                break;
            // Add other actions as needed
            case 'xgboost-prediction':
                showDbNameInput = false; // Show for these actions
                showDbNameSuffix = false; // Show the suffix for these actions
                break;
            case 'prophet-predict':
                showDbNameInput = false; // Show for these actions
                showDbNameSuffix = false; // Show the suffix for these actions
                break;
            case 'ensemble':
                showDbNameInput = false; // Show for these actions
                showDbNameSuffix = false; // Show the suffix for these actions
                break;
            case 'ensemble_prediction':
                showDbNameInput = false; // Show for these actions
                showDbNameSuffix = false; // Show the suffix for these actions
                break;
        }

        // Show or hide the db_name container based on the action
        document.getElementById('db_name_container').style.display = showDbNameInput ? 'block' : 'none';
        document.getElementById('dbNameSuffix').style.display = showDbNameSuffix ? 'block' : 'none'; // Directly control visibility

        // Update the content of dbNameSuffixContainer to display the suffix
        dbNameSuffixContainer.textContent = dbNameSuffix;

        // If the action is "evaluate", call listUserDatabasesWithSuffix
        if(action === 'evaluate' || action === 'predict-next' || action === 'prophet-evaluate' || action === 'xgboost-evaluate' || action === 'xgboost-prediction' || action === 'prophet-predict' || action === 'ensemble' || action === 'ensemble_prediction') {
            listUserDatabasesWithSuffix(dbNameSuffix);
        }

        currentDbNameSuffix = dbNameSuffix;

        userForm.style.display = 'block';
        // Include 'prophet' in the condition for displaying the upload form
        uploadForm.style.display = (action === 'train' || action === 'prophet' || action === 'xgboost') ? 'block' : 'none';

        downloadPlotBtn.style.display = (action === 'prophet' || action === 'xgboost' || action === 'train' || action === 'evaluate' || action === 'xgboost-prediction'  || action == 'xgboost-evaluate' || action == 'prophet-evaluate' || action === 'prophet-predict' || action === 'ensemble' || action === 'ensemble_prediction') ? 'block' : 'none';

        downloadCsvBtn.style.display = (action === 'xgboost-prediction' || action === 'prophet-predict' || action === 'ensemble_prediction') ? 'block' : 'none';

        downloadFeatureImportanceBtn.style.display = (action === 'xgboost' || action == 'xgboost-evaluate') ? 'block' : 'none';

        // Update display and message for 'prophet'
        trainingStatusDisplay.style.display = 'block';
        if (action === 'predict-next') {
            statusMessage.innerHTML = 'Waiting to predict the next movement and price...';
            action_message_Message.innerHTML =  'Please click on the model you want to evaluate then click Go and wait for the results'
        }
        else if (action === 'evaluate') {
            statusMessage.innerHTML = 'Waiting for evaluation to start...';
            action_message_Message.innerHTML =  'Please click on the model you want to evaluate then click Go and wait for the results'
        }
        else if (action === 'xgboost-evaluate') {
            statusMessage.innerHTML = 'Waiting for xgboost evaluation to start...';
            action_message_Message.innerHTML =  'Please click on the model you want to evaluate then click Go and wait for the results'

        }
        else if (action === 'prophet-evaluate') {
            statusMessage.innerHTML = 'Waiting for prophet evaluation to start...';
            action_message_Message.innerHTML =  'Please click on the model you want to evaluate then click Go and wait for the results'

        }
        else if (action === 'train') {
            statusMessage.innerHTML = 'Waiting for training to start...';
            action_message_Message.innerHTML =  'Please enter a name for the model and make sure to upload the a csv with the data.<br>Make sure that the csv has 2 columns.<br>The First coloumn show be named "date" and the second should be named "sales_data_current"<br>Then Click Go and wait for the results'
        }
        else if (action === 'prophet') {
            statusMessage.innerHTML = 'Waiting for prophet training to start...';
            action_message_Message.innerHTML =  'Please enter a name for the model and make sure to upload the a csv with the data.<br>Make sure that the csv has 2 columns.<br>The First coloumn show be named "date" and the second should be named "sales_data_current"<br>Then Click Go and wait for the results'
        }
        else if (action === 'xgboost') {
            statusMessage.innerHTML = 'Waiting for xgboost training to start...';
            action_message_Message.innerHTML =  'Please enter a name for the model and make sure to upload the a csv with the data.<br>Make sure that the csv has 2 columns.<br>The First coloumn show be named "date" and the second should be named "sales_data_current"<br>Then Click Go and wait for the results'
        }
        else if (action === 'xgboost-prediction') {
            statusMessage.innerHTML = 'Waiting for xgboost prediction to start (Prediction Based Of Trained XGboost Data)...';
            action_message_Message.innerHTML =  'Please click on the model you want to evaluate then click Go and wait for the results'
        }
        else if (action === 'prophet-predict') {
            statusMessage.innerHTML = 'Waiting for prophet prediction to start(Prediction Based Of Trained Prophet Data)...';
            action_message_Message.innerHTML =  'Please click on the model you want to evaluate then click Go and wait for the results'
        }
        else if (action === 'ensemble') {
            statusMessage.innerHTML = 'Waiting for ensemble training to start...';
            action_message_Message.innerHTML =  'Please choose the models you want to train and then click go'
        }
        else if (action === 'ensemble_prediction') {
            statusMessage.innerHTML = 'Waiting for ensemble prediction training to start...';
            action_message_Message.innerHTML =  'Please choose the models you want to predict the future sales with and then click go (Prediction Based Of Trained Prophet and XGboost Data)'
        }
    }

    // If an action parameter is present in the URL, update the select dropdown and adjust visibility
    if (actionParam) {
        document.getElementById('action').value = actionParam;
    }

    // Adjust visibility based on the current or updated action
    handleActionChange();

    // Listen for changes in the action dropdown to adjust visibility and reload the page with the new action
    document.getElementById('action').addEventListener('change', function() {
        var action = document.getElementById('action').value;
        // Reload the page with the selected action as a URL parameter
        window.location.href = `?action=${action}`;
    });

    // Ensure handleActionChange is called on page load and when the action changes
    document.addEventListener("DOMContentLoaded", handleActionChange);
    document.getElementById('action').addEventListener('change', handleActionChange);

    updateTrialCountdown()
});

    document.getElementById('logoutButton').addEventListener('click', function() {
        window.location.href = '/logout'; // Redirect to the logout route
    });

    document.getElementById('checkout-button').addEventListener('click', () => {
    const userId = localStorage.getItem('userId');  // Retrieve user_id from local storage

    // Include user_id in the request to your backend (adjust as needed)
    fetch('/create-checkout-session', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ user_id: userId }),
    })
    .then(response => response.json())
    .then(session => {
        window.location.href = session.url;  // Redirect to Stripe Checkout
    })
    .catch(error => console.error('Error:', error));
    });

// Function to update countdown timer for trial or subscription
function updateTrialCountdown() {
    fetch('/get-user-details')
        .then(response => response.json())
        .then(data => {
            const countdownContainer = document.getElementById("trialCountdown");
            const countdownTimer = document.getElementById("countdownTimer");
            const subscriptionButton = document.getElementById("checkout-button");
            const cancelSubscriptionButton = document.getElementById("cancelSubscriptionBtn");
            const reactivateSubscriptionBtn = document.getElementById("reactivateSubscriptionBtn");

            if (data.success) {
                // Determine whether to show trial end date or subscription time
                let endDate;
                let messagePrefix;
                if (data.has_paid && data.subscription_end_date_formatted) {
                    subscriptionButton.style.display = "none";
                    cancelSubscriptionButton.style.display = "none";
                    // User has a subscription
                    endDate = new Date(data.subscription_end_date_formatted).getTime();
                    messagePrefix = "Your subscription ends in: ";
                }
                else if (data.has_paid && data.subscription_end_date_formatted === null){
                    subscriptionButton.style.display = "none";
                    countdownTimer.style.display = "none";
                    reactivateSubscriptionBtn.style.display = "none";
                }
                else {
                    cancelSubscriptionButton.style.display = "none";
                    reactivateSubscriptionBtn.style.display = "none";
                    // User is on a free trial
                    endDate = new Date(data.trial_end_date).getTime();
                    messagePrefix = "Your free trial ends in: ";
                }

                const x = setInterval(() => {
                    const now = new Date().getTime();
                    const distance = endDate - now;

                    const days = Math.floor(distance / (1000 * 60 * 60 * 24));
                    const hours = Math.floor((distance % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
                    const minutes = Math.floor((distance % (1000 * 60 * 60)) / (1000 * 60));
                    const seconds = Math.floor((distance % (1000 * 60)) / 1000);

                    countdownTimer.innerHTML = `${messagePrefix} ${days}d ${hours}h ${minutes}m ${seconds}s `;

                    if (distance < 0) {
                        clearInterval(x);
                        countdownTimer.innerHTML = "EXPIRED";
                    }
                }, 1000);
            } else {
                console.error('Failed to fetch user details:', data.message);
                countdownTimer.innerHTML = "Error loading trial time.";
            }
        })
        .catch(error => {
            console.error('Error fetching user details:', error);
            countdownTimer.innerHTML = "Error loading trial time.";
        });
}

document.getElementById('reactivateSubscriptionBtn').addEventListener('click', function() {
    // Fetch the current user's username from somewhere, e.g., local storage, or embed in HTML
    const username = localStorage.getItem('userId'); // Example: fetching from local storage

    // Make a fetch request to your Flask route for subscription cancellation
    fetch('/reactivate-subscription', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username: username }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Handle successful cancellation (e.g., display a message, redirect, etc.)
            window.location.reload();
        } else {
            // Handle errors
            alert(data.error);
        }
    })
    .catch(error => console.error('Error:', error));
});

document.getElementById('cancelSubscriptionBtn').addEventListener('click', function() {
    // Fetch the current user's username from somewhere, e.g., local storage, or embed in HTML
    const username = localStorage.getItem('userId'); // Example: fetching from local storage

    // Make a fetch request to your Flask route for subscription cancellation
    fetch('/cancel-subscription', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username: username }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Handle successful cancellation (e.g., display a message, redirect, etc.)
            window.location.reload();
        } else {
            // Handle errors
            alert(data.error);
        }
    })
    .catch(error => console.error('Error:', error));
});

function setUserAndDB() {
    var userId = localStorage.getItem('userId');
    var dbName = document.getElementById('db_name').value;
    var action = document.getElementById('action').value;

    var dbNameWithSuffix = dbName + currentDbNameSuffix; // Automatically add the suffix

    if (action === 'prophet' || action === 'xgboost' || action === 'ensemble' || action === 'train') {
        document.getElementById('status').innerHTML = 'Starting Training...';
    } else if (action === 'prophet-evaluate' || action === 'xgboost-evaluate' || action === 'evaluate') {
        document.getElementById('status').innerHTML = 'Starting Evaluation...';
    } else if (action === 'prophet-predict' || action === 'xgboost-prediction' || action === 'ensemble_prediction') {
        document.getElementById('status').innerHTML = 'Getting Prediction...';
    }

    if (document.getElementById('action').value === 'prophet' || document.getElementById('action').value === 'xgboost' || document.getElementById('action').value === 'xgboost-prediction' || document.getElementById('action').value === 'prophet-predict' || document.getElementById('action').value === 'ensemble' || document.getElementById('action').value === 'ensemble_prediction' || document.getElementById('action').value === 'train'){
        uploadFile();
    }
    else if (document.getElementById('action').value === 'evaluate' || document.getElementById('action').value === 'xgboost-evaluate' || document.getElementById('action').value === 'prophet-evaluate' || document.getElementById('action').value === 'predict-next') {
        startScript();
    }
}

function createDeleteIcon(dbName) {
    const deleteIcon = document.createElement('span');
    deleteIcon.innerHTML = '🗑️'; // Emoji as a placeholder; consider using an SVG or icon font in production
    deleteIcon.classList.add('delete-db-icon', 'ml-2', 'text-red-500', 'hover:text-red-700', 'cursor-pointer');
    deleteIcon.dataset.dbName = dbName;
    deleteIcon.title = "Delete Database";

    // Confirm deletion with the user
    deleteIcon.addEventListener('click', function(event) {
        event.stopPropagation(); // Prevent click from reaching the dbElement
        const isConfirmed = confirm(`Are you sure you want to delete ${dbName}?`);
        if (isConfirmed) {
            // Call function to handle deletion
            deleteDatabase(dbName);
        }
    });

    return deleteIcon;
}


function listUserDatabasesWithSuffix(suffix) {
    var userId = localStorage.getItem('userId');
    var action = document.getElementById('action').value;
    if (!userId) {
        console.error('User ID not found in localStorage.');
        return;
    }

    fetch('/list-dbs', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            user_id: userId,
            suffix: suffix,
            action: action
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.databases && data.databases.length > 0) {
            console.log('Filtered Databases:', data.databases);
            const container = document.getElementById('databaseListContainer');
            container.innerHTML = '';
            container.classList.add('flex', 'flex-col', 'gap-2'); // Tailwind CSS for spacing

            // Adjusted syntax error
            if (action !== 'ensemble' && action !== 'ensemble_prediction') {
                // Single selection logic
                data.databases.forEach(dbName => {
                    // Create a div to hold both the database name and the delete icon
                    const dbContainer = document.createElement('div');
                     dbContainer.classList.add('db-container', 'flex', 'justify-between', 'items-center', 'p-2', 'bg-gray-200', 'rounded', 'hover:bg-gray-300', 'cursor-pointer');


                    // Create an element for the database name
                    const dbElement = document.createElement('span');
                    dbElement.classList.add('db-name', 'text-gray-800', 'text-sm', 'font-semibold');
                    dbElement.textContent = dbName;
                    dbElement.type = 'button';
                    dbElement.classList.add('db-list-item');
                    // Assuming this is inside the forEach loop for single selection logic
                    dbElement.addEventListener('click', function(event) {
                        event.preventDefault();

                        // Remove selected class from any previously selected container
                        const previouslySelected = document.querySelector('.db-container.db-selected');
                        if (previouslySelected) {
                            previouslySelected.classList.remove('db-selected', 'bg-blue-500', 'text-white');
                        }

                        // Add selected class to the current container
                        dbContainer.classList.add('db-selected', 'bg-blue-500', 'text-white');

                        // Update the input value
                        document.getElementById('db_name').value = dbName.replace('.db', '');
                    });
                    container.appendChild(dbElement);

                dbElement.classList.add('db-list-item');
                dbElement.textContent = dbName;


                // Create and append the delete icon
                const deleteIcon = createDeleteIcon(dbName);

                // Append the delete icon to each database element
                // Append both the database name and delete icon to the container
                dbContainer.appendChild(dbElement);
                dbContainer.appendChild(deleteIcon);

                container.appendChild(dbContainer);
                });
            } else {
                data.databases.forEach(dbName => {
                    // Create a div to hold both the database name and the delete icon
                    const dbContainer = document.createElement('div');
                     dbContainer.classList.add('db-container', 'flex', 'justify-between', 'items-center', 'p-2', 'bg-gray-200', 'rounded', 'hover:bg-gray-300', 'cursor-pointer');

                    // Create an element for the database name
                    const dbElement = document.createElement('span');
                    dbElement.classList.add('db-name', 'text-gray-800', 'text-sm', 'font-semibold');
                    dbElement.textContent = dbName;

                    // Assuming this is inside the forEach loop for multiple selection logic
                    dbElement.addEventListener('click', function(event) {
                        event.preventDefault(); // Prevent default to avoid any unintended page navigation

                        // Toggle 'db-selected' class for visual feedback on the container
                        if (dbContainer.classList.contains('db-selected')) {
                            // It's already selected, remove the classes
                            dbContainer.classList.remove('db-selected', 'bg-blue-500', 'text-white');
                        } else {
                            // It's not selected, add the classes
                            dbContainer.classList.add('db-selected', 'bg-blue-500', 'text-white');
                        }

                        // Update the value of the input for selected databases
                        // We ensure to select '.db-name' from '.db-container.db-selected' to get only the names of selected databases
                        const selectedDbs = document.querySelectorAll('.db-container.db-selected .db-name');
                        const selectedDbNames = Array.from(selectedDbs).map(db => db.textContent.replace('.db', ''));
                        document.getElementById('db_name').value = selectedDbNames.join(', '); // Join the selected database names with a comma
                    });

                    // Append the database name element to the container
                    dbContainer.appendChild(dbElement);

                    // Create and append the delete icon
                    const deleteIcon = createDeleteIcon(dbName);
                    deleteIcon.addEventListener('click', function(event) {
                        event.stopPropagation(); // Stop the click from propagating to dbElement
                    });

                    // Append the delete icon to the container
                    dbContainer.appendChild(deleteIcon);

                    // Finally, append the container to the parent container in the DOM
                    container.appendChild(dbContainer);
                });
            }
        } else {
            console.error('No databases found or error fetching databases.');
            document.getElementById('databaseListContainer').innerHTML = '<p>No Models Found</p>';
        }
    })
    .catch(error => console.error('Error fetching databases:', error));
}

// Assuming you have a deletion endpoint set up, here's the deleteDatabase function:
function deleteDatabase(dbName) {
    const userId = localStorage.getItem('userId');
    // Assuming '/delete-db' is your endpoint for deleting databases
    fetch('/delete-db', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            user_id: userId,
            db_name: dbName
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            listUserDatabasesWithSuffix(currentDbNameSuffix); // Refresh the list
        } else {
            alert('Failed to delete database.');
        }
    })
    .catch(error => console.error('Error deleting database:', error));
}

const fileInput = document.getElementById('file');

function uploadFile() {
    var userId = localStorage.getItem('userId');
    var dbName = document.getElementById('db_name').value;
    var fileInput = document.getElementById('file');
    var action = document.getElementById('action').value;
    var formData = new FormData();

    if (action !== 'prophet' || action !=='xgboost' || action !== 'train'){
        var dbName = document.getElementById('db_name').value;
        var dbNameWithSuffix = dbName + currentDbNameSuffix;
    }
    if (action === 'prophet' || action === 'xgboost' || action === 'train'){
        const fileName = fileInput.files[0].name;  // Access fileName from event listener
        const fileNameWithoutExtension = fileName.substring(0, fileName.lastIndexOf('.'));  // Get everything before the last dot
        var dbName = fileNameWithoutExtension;
        var dbNameWithSuffix = dbName + currentDbNameSuffix;
    }

    formData.append('file', fileInput.files[0]);
    formData.append('user_id', userId); // Add userId to the FormData
    formData.append('db_name', dbNameWithSuffix)
    formData.append('action', action)

    let upload_successful = false;

    // Reset uploadRetryCount if starting a new upload process
    uploadRetryCount = 0;

    // Define a function to attempt the file upload
    function attemptUpload() {
        fetch('/upload-csv', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            console.log(data);
            if (data.message !== 'File not allowed') {
                let upload_successful = true;
                uploadRetryCount = 0; // Reset retry counter upon success
                startScript(); // Continue to the next step if the upload is successful
            } else {
                document.getElementById('status').innerHTML = `Error. Please upload a valid CSV File`
            }
        })
        .catch(error => {
            console.error('Error:', error);
            retryUploadIfNeeded();
        });
    }

    // Define a function to handle retries
    function retryUploadIfNeeded() {
        if (uploadRetryCount < uploadMaxRetries) {
            uploadRetryCount++;
            console.log(`Retrying upload... Attempt ${uploadRetryCount}`);
            setTimeout(attemptUpload, 3000); // Wait for 3 seconds before retrying
        } else {
            console.error("Maximum retry attempts for upload reached. Please check the issue and try again later.");
        }
    }

    // Start the upload attempt
    attemptUpload();
}

function fetchLatestTaskStatus() {
    var userId = localStorage.getItem('userId'); // Assuming user ID is stored in local storage
    if (!userId) {
        console.error('User ID not found in local storage.');
        alert('User ID is required to fetch the latest task status.');
        return;
    }

    // Correctly append the user ID as a query parameter
    var url = `/latest-task-status?user_id=${userId}`;

    fetch(url, {
        method: 'GET'
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            console.error('Error fetching latest task status:', data.error);
            alert('Error fetching latest task status. Please try again.');
        } else {
            console.log('Latest task status:', data);
            // Assuming data.result is a JSON string containing an array with two elements
            if (data.result) {
                // Parse the result string into an array
                const resultArray = JSON.parse(data.result);
                const mseError = resultArray[0];
                const accuracyPercentage = resultArray[1];
                // Update the UI with the parsed result
                document.getElementById('status').innerHTML = `Latest Task Status: ${data.status}. MSE Error: ${mseError}, Accuracy: ${accuracyPercentage}%`;
            } else {
                // Handle cases where result might be null or undefined
                document.getElementById('status').innerHTML = `Latest Task Status: ${data.status}. Result: Data not available`;
            }
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Failed to fetch latest task status. Please check console for details.');
    });
}


function downloadPlot() {
    var dbName = document.getElementById('db_name').value;
    var userId = localStorage.getItem('userId');
    var action = document.getElementById('action').value;

    if (action !== 'prophet' || action !=='xgboost' || action !== 'train'){
        var dbName = document.getElementById('db_name').value;
        var dbNameWithSuffix = dbName + currentDbNameSuffix;
    }
    if (action === 'prophet' || action === 'xgboost' || action === 'train'){
        const fileName = fileInput.files[0].name;  // Access fileName from event listener
        const fileNameWithoutExtension = fileName.substring(0, fileName.lastIndexOf('.'));  // Get everything before the last dot
        var dbName = fileNameWithoutExtension;
        var dbNameWithSuffix = dbName + currentDbNameSuffix;
    }

    fetch('/download-plot', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({db_name: dbNameWithSuffix, user_id: userId, action: action}),
    })
    .then(response => response.blob())
    .then(blob => {
        // Create a new URL for the blob
        const url = window.URL.createObjectURL(blob);
        // Create a new link element
        const a = document.createElement('a');
        a.href = url;
        a.download = 'plot.png'; // Set the filename
        document.body.appendChild(a); // Append the link to the body
        a.click(); // Simulate a click on the link
        document.body.removeChild(a); // Remove the link from the body
        window.URL.revokeObjectURL(url); // Clean up the URL object
    })
    .catch(error => {
        console.error('Error downloading the plot:', error);
    });
}

function downloadCSV() {
    var dbName = document.getElementById('db_name').value;
    var userId = localStorage.getItem('userId');
    var action = document.getElementById('action').value; // Assuming you have an action field you want to send

    fetch('/download-predictions', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        // Corrected JSON.stringify syntax
        body: JSON.stringify({user_id: userId, db_name: dbName, action: action}),
    })
    .then(response => response.blob())
    .then(blob => {
        // Create a new URL for the blob
        const url = window.URL.createObjectURL(blob);
        // Create a new link element
        const a = document.createElement('a');
        a.href = url;
        // Changed the filename to indicate it's a CSV file
        a.download = 'predictions.csv';
        document.body.appendChild(a); // Append the link to the body
        a.click(); // Simulate a click on the link
        document.body.removeChild(a); // Remove the link from the body
        window.URL.revokeObjectURL(url); // Clean up the URL object
    })
    .catch(error => {
        console.error('Error downloading the CSV:', error);
    });
}

function downloadFeatureImportance() {
    var dbName = document.getElementById('db_name').value;
    var userId = localStorage.getItem('userId');
    var action = document.getElementById('action').value;

    if (action !== 'prophet' || action !=='xgboost' || action === 'train'){
        var dbName = document.getElementById('db_name').value;
        var dbNameWithSuffix = dbName + currentDbNameSuffix;
    }
    if (action === 'prophet' || action === 'xgboost' || action === 'train'){
        const fileName = fileInput.files[0].name;  // Access fileName from event listener
        const fileNameWithoutExtension = fileName.substring(0, fileName.lastIndexOf('.'));  // Get everything before the last dot
        var dbName = fileNameWithoutExtension;
        var dbNameWithSuffix = dbName + currentDbNameSuffix;
    }

    fetch('/download-feature-importance', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({db_name: dbNameWithSuffix, user_id: userId, action: action}),
    })
    .then(response => response.blob())
    .then(blob => {
        // Create a new URL for the blob
        const url = window.URL.createObjectURL(blob);
        // Create a new link element
        const a = document.createElement('a');
        a.href = url;
        a.download = 'plot.png'; // Set the filename
        document.body.appendChild(a); // Append the link to the body
        a.click(); // Simulate a click on the link
        document.body.removeChild(a); // Remove the link from the body
        window.URL.revokeObjectURL(url); // Clean up the URL object
    })
    .catch(error => {
        console.error('Error downloading the plot:', error);
    });
}

// Initialize a counter for retry attempts at the top of your script
let retryCount = 0;
const maxRetries = 2; // Maximum number of retries

function startScript() {
    var userId = localStorage.getItem('userId');
    var action = document.getElementById('action').value;
    var numDays = (action === 'xgboost-prediction' || action === 'prophet-predict' || action === 'ensemble_prediction') ? document.getElementById('num_days').value : undefined;

    if (action !== 'prophet' || action !=='xgboost' || action !== 'train'){
        var dbName = document.getElementById('db_name').value;
        var dbNameWithSuffix = dbName + currentDbNameSuffix;
    }
    if (action === 'prophet' || action === 'xgboost' || action === 'train'){
        const fileName = fileInput.files[0].name;  // Access fileName from event listener
        const fileNameWithoutExtension = fileName.substring(0, fileName.lastIndexOf('.'));  // Get everything before the last dot
        var dbName = fileNameWithoutExtension;
        var dbNameWithSuffix = dbName + currentDbNameSuffix;
    }

    var requestBody = {
        action: action,
        user_id: userId,
        db_name: dbNameWithSuffix,
        num_days: numDays // Include this only if the action is xgboost-prediction
    };

    // Start the process
    fetch('/run-script', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        if (action === 'train') {
            document.getElementById('status').innerHTML = `Evaluation completed.  MSE Error is ${data.mse_error} and Accuracy is ${data.accuracy}`;
        } else if (action === 'evaluate') {
            // Handle the evaluation response
            const accuracy = data.accuracy ? parseFloat(data.accuracy) : 0;
            const roundedAccuracy = Math.round(accuracy * 100) / 100; // Rounds to 2 decimal places
            document.getElementById('status').innerHTML = `Evaluation completed.  MSE Error is ${data.mse_error} and Accuracy is ${data.accuracy}`;
        } else if (action === 'predict-next') {
            document.getElementById('status').innerHTML = `The last day predicted movement is ${data.last_day_predicted_sales} and next day predicted sales are ${data.next_day_predicted_sales} therefore we predicts your sales will ${data.predicted_movement}`
        } else if (action === 'prophet') {
            document.getElementById('status').innerHTML = `Training completed.<br>Accuracy is ${data.movement_accuracy}% (Sales Movement Accuracy - higher is better)<br>RMSE is ${data.rmse} (Exact error (minimizing large errors is crucial) - lower is better) `;
        } else if (action === 'xgboost') {
            document.getElementById('status').innerHTML = `Training completed.<br>Accuracy is ${data.movement_accuracy}% (Sales Movement Accuracy - higher is better)<br>RMSE is ${data.rmse} (Exact error (minimizing large errors is crucial) - lower is better) `;
        } else if (action === 'xgboost-evaluate') {
            document.getElementById('status').innerHTML = `Evaluation completed.<br>Accuracy is ${data.movement_accuracy}% (Sales Movement Accuracy - higher is better)<br>RMSE is ${data.rmse} (Exact error (minimizing large errors is crucial) - lower is better) `;
        } else if (action === 'prophet-evaluate') {
            document.getElementById('status').innerHTML = `Evaluation completed.<br>Accuracy is ${data.movement_accuracy}% (Sales Movement Accuracy - higher is better)<br>RMSE is ${data.rmse} (Exact error (minimizing large errors is crucial) - lower is better) `;
        } else if (action === 'xgboost-prediction') {
            document.getElementById('status').innerHTML = `Please download the prediction graph below`;
        } else if (action === 'prophet-predict') {
            document.getElementById('status').innerHTML = `Please download the prediction graph below`;
        } else if (action === 'ensemble') {
            document.getElementById('status').innerHTML = `Please download the prediction graph below.<br>Accuracy is ${data.movement_accuracy}% (Sales Movement Accuracy - higher is better)<br>RMSE is ${data.rmse} (Exact error (minimizing large errors is crucial) - lower is better)`;
        } else if (action === 'ensemble_prediction') {
            document.getElementById('status').innerHTML = `Please download the prediction graph below`;
        } else {
            document.getElementById('status').innerHTML = 'Operation completed. Check the console for more details.';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('status').innerHTML = 'This csv file needs more time. Please contact us via the contact form and will assist you further';
    });
}