// Helper function to convert FormData to JSON
function formDataToJson(formData) {
    const object = {};
    formData.forEach((value, key) => object[key] = value);
    return JSON.stringify(object);
}

// Handle the signup form submission
document.getElementById('signupForm').addEventListener('submit', function(e) {
    e.preventDefault(); // Prevent default form submission
    const formData = new FormData(this);
    const jsonData = formDataToJson(formData);

    fetch('/signup', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: jsonData
    }).then(response => response.json()) // Convert the response to JSON
    .then(data => {
        alert(data.message); // Display the message from the server
        if (data.success) {
            // Optional: Clear the form fields after successful signup
            this.reset();
        }
    })
    .catch(error => {
        console.error('There has been a problem with your fetch operation:', error);
        alert('Signup failed due to a network error.');
    });
});

// This should be defined somewhere in your script
function formDataToJson(formData) {
    const object = {};
    formData.forEach((value, key) => { object[key] = value; });
    return JSON.stringify(object);
}

document.getElementById('loginForm').addEventListener('submit', function(e) {
    e.preventDefault();

    const formData = new FormData(this);
    const jsonData = formDataToJson(formData);

    const userId = document.getElementById('user_id').value.trim();

    fetch('/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: jsonData
    })
    .then(response => {
        if (!response.ok) {
            // If the server response was not OK, interpret the response body to get the error message
            return response.json().then(data => Promise.reject(data.message));
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            localStorage.setItem('userId', userId);
            window.location.href = '/main'; // Adjust as needed
        }
    })
    .catch(error => {
        console.error('There has been a problem with your fetch operation:', error);
        alert('Login failed: ' + error);
    });
});

document.getElementById('darkThemeButtonForgotPasswordBtn').addEventListener('click', function() {
    var email = prompt("Please enter your email address:");
    var new_password = prompt("Please enter your new password:");

    fetch('/reset-password', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            email: email,
            new_password: new_password
        }),
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
    })
    .catch((error) => {
        console.error('Error:', error);
    });
});