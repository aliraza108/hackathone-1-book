import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

type FeatureItem = {
  title: string;
  description: JSX.Element;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Physical AI & Embodied Intelligence',
    description: (
      <>
        Learn about the principles of Physical AI and how embodiment shapes intelligence in robotic systems.
      </>
    ),
  },
  {
    title: 'ROS 2 & Modern Robotics Framework',
    description: (
      <>
        Master ROS 2 architecture, nodes, topics, and services for building complex robotic applications.
      </>
    ),
  },
  {
    title: 'Humanoid Robotics & Control',
    description: (
      <>
        Explore humanoid kinematics, dynamics, and locomotion control for bipedal robots.
      </>
    ),
  },
];

function Feature({title, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): JSX.Element {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}