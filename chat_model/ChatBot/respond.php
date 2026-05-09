<?php
header('Content-Type: application/json');

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode(['error' => 'Method not allowed']);
    exit;
}

$input = json_decode(file_get_contents('php://input'), true);
$message = isset($input['message']) ? trim($input['message']) : '';

if ($message === '') {
    http_response_code(400);
    echo json_encode(['error' => 'Message is required']);
    exit;
}

$pythonScript = __DIR__ . '/main.py';
$command = 'python3 ' . escapeshellarg($pythonScript) . ' ' . escapeshellarg($message) . ' 2>&1';
$output = shell_exec($command);

if ($output === null) {
    http_response_code(500);
    echo json_encode(['error' => 'Failed to run model']);
    exit;
}

// Try to parse as JSON, but if it fails, wrap the output in an error response
$decoded = @json_decode($output, true);
if ($decoded === null) {
    // Output is not valid JSON - likely an error message or crash
    http_response_code(500);
    echo json_encode(['error' => 'Model error: ' . trim($output)]);
    exit;
}

if (isset($decoded['error'])) {
    http_response_code(500);
}

echo $output;