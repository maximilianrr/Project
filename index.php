
<!Doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MedChat</title>
    <link rel="stylesheet" href="style.css">
</head>

<body>
    <h1>MedChat</h1>
    <div class="container">
        <div class="chat-box" id="chat-box">
            <!-- Chat messages will appear here -->
             <div class="model-message">
                <div class="model-message-content" id="model-message-content">
                    <p>Hi! I'm MedChat, your medical assistant. How can I help you today?</p>
                </div>
             </div>
        </div>
        <form id="chat-form">
            <textarea id="user-input" placeholder="Type your message..." rows="1" required></textarea>
            <button type="submit" aria-label="Send message">→</button>
        </form>
    </div>

    <script src="script.js"></script>