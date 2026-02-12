import React from 'react';
import styles from './styles.module.css';

const LoadingSpinner = ({ size = 'medium', message = 'Processing...' }) => {
  const sizeClass = styles[`spinner-${size}`];
  
  return (
    <div className={styles.spinnerContainer}>
      <div className={`${styles.spinner} ${sizeClass} ${styles.crtBorder}`}>
        <div className={styles.spinnerCircle}></div>
        <div className={styles.spinnerCircle}></div>
        <div className={styles.spinnerCircle}></div>
      </div>
      <div className={styles.spinnerMessage}>{message}</div>
    </div>
  );
};

export default LoadingSpinner;