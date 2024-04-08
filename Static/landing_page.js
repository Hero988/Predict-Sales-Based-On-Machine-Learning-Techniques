document.getElementById('loginButton').addEventListener('click', function() {
    window.location.href = '/login-page'; // Redirects the user to the login page
});

document.getElementById('loginButton2').addEventListener('click', function() {
window.location.href = '/login-page'; // Redirects the user to the login page
});

document.getElementById('loginButton3').addEventListener('click', function() {
window.location.href = '/login-page'; // Redirects the user to the login page
});

document.getElementById('emailButton').addEventListener('click', function(e) {
  e.preventDefault(); // Prevent form from submitting traditionally

  const emailInput = document.querySelector('input[type="email"]');
  const messageInput = document.querySelector('textarea');

  const email = emailInput.value;
  const message = messageInput.value;

  // Simple validation
  if (!email || !message) {
    alert('Please fill in both email and message.');
    return;
  }

  fetch('/send-email', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({email: email, message: message}),
  })
  .then(response => response.json())
  .then(data => {
    alert(data.message); // Inform the user
    emailInput.value = ''; // Reset
    messageInput.value = ''; // Reset
  })
  .catch(error => console.error('Error:', error)); // Catch and log any errors
});