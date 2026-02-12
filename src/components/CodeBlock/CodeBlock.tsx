import React from 'react';
import styles from './styles.module.css';

const CodeBlock = ({ children, language = '', title = '' }) => {
  return (
    <div className={`${styles.codeBlock} ${styles.crtBorder} ${styles.crtScanlines}`}>
      {title && <div className={styles.codeTitle}>{title}</div>}
      <pre className={styles.preFormatted}>
        <code className={styles.codeContent}>
          {children}
        </code>
      </pre>
    </div>
  );
};

export default CodeBlock;