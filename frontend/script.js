/**
 * Frontend JavaScript
 * 
 * Handles chat UI interactions and API communication.
 * NO API keys or memory logic here - all handled by backend.
 */

// Use relative URL since frontend is served from same origin as backend
const API_URL = '/chat';

const messagesContainer = document.getElementById('messages');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');

/**
 * Add a message to the chat UI
 */
function addMessage(text, isUser) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.textContent = text;
    
    messageDiv.appendChild(bubble);
    messagesContainer.appendChild(messageDiv);
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

/**
 * Show error message
 */
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = `Error: ${message}`;
    messagesContainer.appendChild(errorDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

/**
 * Send message to backend
 */
async function sendMessage() {
    const message = userInput.value.trim();
    
    if (!message) {
        return;
    }
    
    // Disable input while processing
    userInput.disabled = true;
    sendButton.disabled = true;
    sendButton.innerHTML = '<div class="loading"></div>';
    
    // Show user message
    addMessage(message, true);
    userInput.value = '';
    
    try {
        // Send POST request to backend
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || `HTTP error! status: ${response.status}`);
        }
        
        // Show assistant reply
        addMessage(data.reply, false);
        
    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'Failed to get response. Make sure the backend is running.');
    } finally {
        // Re-enable input
        userInput.disabled = false;
        sendButton.disabled = false;
        sendButton.textContent = 'Send';
        userInput.focus();
    }
}

// Event listeners
sendButton.addEventListener('click', sendMessage);

userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Focus input on load
userInput.focus();
