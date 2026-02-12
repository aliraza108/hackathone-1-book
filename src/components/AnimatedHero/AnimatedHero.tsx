import React, { useState, useEffect } from 'react';
import styles from './styles.module.css';

const AnimatedHero = () => {
  const [displayText, setDisplayText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);
  
  const heroText = `
    ╔══════════════════════════════════════╗
    ║  ▄▄▄▄· ▄▄▄  ·▄▄▄▄▄▄▄▄  ·▄▄▄▄  ▄▄▄ .  ║
    ║  ▐█ ▀█▪▀▄ █·██• ██  ▀▄.▀·▐▄▄· ▀▄.▀·  ║
    ║  ▐█▀▀█▄▐▀▀▄ ▐█·▐█.▪▐▀▀▪▄██▪  ▐▀▀▪▄  ║
    ║  ██▄▪▐█▐█•█▌▐█▌▐█▌·▐█▄▄▌██▌. ▐█▄▄▌  ║
    ║  ·▀▀▀▀ .▀  ▀▀▀▀▀▀▀  ▀▀▀ ▀▀▀   ▀▀▀   ║
    ║                                      ║
    ║    PHYSICAL AI & HUMANOID ROBOTICS   ║
    ║         TEXTBOOK COURSE              ║
    ╚══════════════════════════════════════╝
  `;

  useEffect(() => {
    if (currentIndex < heroText.length) {
      const timeout = setTimeout(() => {
        setDisplayText(prev => prev + heroText[currentIndex]);
        setCurrentIndex(prev => prev + 1);
      }, 30); // Adjust typing speed here

      return () => clearTimeout(timeout);
    }
  }, [currentIndex, heroText]);

  return (
    <div className={styles.heroContainer}>
      <pre className={styles.asciiArt}>{displayText}<span className={styles.cursor}></span></pre>
    </div>
  );
};

export default AnimatedHero;