import React from 'react';
import styles from './styles.module.css';

const RetroButton = ({ children, variant = 'primary', size = 'medium', onClick, disabled = false, ...props }) => {
  const buttonClasses = [
    styles.retroButton,
    styles[`variant-${variant}`],
    styles[`size-${size}`],
    disabled ? styles.disabled : ''
  ].filter(Boolean).join(' ');

  return (
    <button 
      className={buttonClasses} 
      onClick={onClick} 
      disabled={disabled}
      {...props}
    >
      {children}
    </button>
  );
};

export default RetroButton;