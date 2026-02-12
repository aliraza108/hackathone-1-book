import React from 'react';
import useChatbot from '../../hooks/useChatbot';
import styles from './styles.module.css';

const ChatWidget = () => {
  const {
    messages,
    input,
    setInput,
    sendMessage,
    isLoading,
    isWidgetOpen,
    toggleWidget
  } = useChatbot();

  // Suggested prompts for the user
  const suggestedPrompts = [
    "What is Physical AI?",
    "Explain ROS 2 architecture",
    "How do humanoid robots maintain balance?",
    "What are the challenges in humanoid robotics?"
  ];

  const handleSuggestedPrompt = (prompt: string) => {
    setInput(prompt);
    sendMessage(); // Automatically send the suggested prompt
  };

  if (!isWidgetOpen) {
    return (
      <button
        className={styles.chatLauncher}
        onClick={toggleWidget}
        aria-label="Open chat"
      >
        💬
      </button>
    );
  }

  return (
    <div className={styles.chatContainer}>
      <div className={styles.chatHeader}>
        <h3>AI Assistant</h3>
        <button onClick={toggleWidget} aria-label="Close chat">
          ×
        </button>
      </div>
      <div className={styles.chatMessages}>
        {messages.length === 0 ? (
          <div className={styles.welcomeMessage}>
            <div className={`${styles.message} ${styles.bot}`}>
              Hello! I'm your AI assistant for Physical AI & Humanoid Robotics. How can I help you today?
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg, index) => (
              <div 
                key={index} 
                className={`${styles.message} ${styles[msg.sender]}`}
              >
                {msg.text}
              </div>
            ))}
            {isLoading && (
              <div className={`${styles.message} ${styles.bot}`}>
                <div className={styles.typingIndicator}>
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* Suggested prompts section */}
      {messages.length === 0 && !isLoading && (
        <div className={styles.suggestedPrompts}>
          {suggestedPrompts.map((prompt, index) => (
            <button
              key={index}
              className={styles.suggestedPrompt}
              onClick={() => handleSuggestedPrompt(prompt)}
            >
              {prompt}
            </button>
          ))}
        </div>
      )}

      <div className={styles.chatInputArea}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              sendMessage();
            }
          }}
          placeholder="Ask about Physical AI, ROS 2, Humanoid Robotics..."
          rows={1}
          aria-label="Type your message"
        />
        <button 
          onClick={sendMessage} 
          disabled={isLoading || !input.trim()}
          aria-label="Send message"
        >
          Send
        </button>
      </div>
    </div>
  );
};

export default ChatWidget;