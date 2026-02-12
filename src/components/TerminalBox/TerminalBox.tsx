import React from 'react';
import styles from './styles.module.css';

const TerminalBox = ({ children, title, ...props }) => {
  return (
    <div className={`${styles.terminalBox} ${styles.crtBorder}`} {...props}>
      {title && <div className={styles.terminalTitle}>{title}</div>}
      <div className={styles.terminalContent}>{children}</div>
    </div>
  );
};

export default TerminalBox;