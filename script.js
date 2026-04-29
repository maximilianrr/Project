// height adjusment for the input field, with a max height of 180px. If the content exceeds this height, a scrollbar will appear. Additionally, the border radius of the input field changes based on its height to maintain a visually appealing design.
document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('chat-form');
  const input = document.getElementById('user-input');
  const maxHeight = 180;

  if (!form || !input) {
    return;
  }

  const resizeInput = () => {
    input.style.height = 'auto';
    const nextHeight = Math.min(input.scrollHeight, maxHeight);
    input.style.height = `${nextHeight}px`;
    input.style.overflowY = input.scrollHeight > maxHeight ? 'auto' : 'hidden';
    if (nextHeight > 64) {
      input.style.borderRadius = '20px';
    } else { 
        input.style.borderRadius = '45px';
    }
  };

  // Event listener for input changes to resize the input field dynamically
  input.addEventListener('input', resizeInput);
  form.addEventListener('submit', (event) => {
    event.preventDefault();

    const message = input.value.trim();
    if (!message) {
      return;
    }

    // Add the user's message to the chat box and clear the input field
    addChatMessage(message, 'user');
    input.value = '';
    resizeInput();
    sendMessageToModel(message);
  });

  resizeInput();
});

// Function to send the user's message to the backend model and handle the response. It sends a POST request to 'chat_model/respond.php' with the user's message in JSON format. Upon receiving a response, it adds the model's reply to the chat box. If there's an error during the fetch operation, it logs the error to the console.
function sendMessageToModel(message) {
    if (message.trim() === '') {
        return;
    } else {
        fetch('chat_model/respond.php', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message })
        })
            .then(response => response.json())
            .then(data => {
                addChatMessage(data.response, 'model');
            })
            .catch(error => {
                console.error('Error:', error);
        });
    }
}

// Function to add a chat message to the chat box. It creates a new message element based on the sender (user or model) and appends it to the chat box. The chat box is then scrolled to the bottom to ensure the latest message is visible.
function addChatMessage(message, sender) {
    const chatBox = document.getElementById('chat-box');
    const messageElement = document.createElement('div');
    const contentElement = document.createElement('div');

    if (sender === 'user') {
        messageElement.classList.add('sent-message');
        contentElement.classList.add('sent-message-content');
    } else {
        messageElement.classList.add('model-message');
        contentElement.classList.add('model-message-content');
    }

    contentElement.textContent = message;
    messageElement.appendChild(contentElement);
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;
}
