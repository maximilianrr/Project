
<!Doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Med Chat</title>
    <link rel="stylesheet" href="style.css">
</head>

<body>
    <h1>Med Chat</h1>
    <div class="container">
        <div class="chat-box" id="chat-box">
            <!-- Chat messages will appear here -->
        </div>
        <form id="chat-form">
            <input type="text" id="user-input" placeholder="Type your message..." required>
            <button type="submit" aria-label="Send message">→</button>
        </form>
    </div>

    <script src="script.js"></script>