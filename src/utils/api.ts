// API utility functions for the chatbot
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const sendMessage = async (message: string, conversationId?: string) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/chat/message`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        conversation_id: conversationId || null
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error sending message:', error);
    throw error;
  }
};

export const sendSelectionQuery = async (selectedText: string, context: string, conversationId?: string) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/chat/selection`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        selected_text: selectedText,
        context,
        conversation_id: conversationId || null
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error sending selection query:', error);
    throw error;
  }
};

export const getChatHistory = async (conversationId: string) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/chat/history?conversation_id=${conversationId}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error getting chat history:', error);
    throw error;
  }
};