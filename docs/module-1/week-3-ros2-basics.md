---
sidebar_position: 3
---

# Week 3: ROS 2 Architecture & Core Concepts

## Learning Objectives

By the end of this week, you will be able to:
- Explain the architecture of ROS 2 and its key components
- Understand the differences between ROS 1 and ROS 2
- Describe the DDS (Data Distribution Service) middleware
- Implement basic ROS 2 nodes and topics
- Use ROS 2 tools for debugging and visualization

## Introduction to ROS 2

Robot Operating System 2 (ROS 2) is the next-generation robotics framework designed to address the limitations of ROS 1 and meet the requirements of commercial and industrial robotics applications. Unlike ROS 1, which relied on a centralized master architecture, ROS 2 uses a decentralized architecture based on DDS (Data Distribution Service).

### Key Improvements in ROS 2

1. **Real-time support**: Deterministic behavior for time-critical applications
2. **Multi-robot systems**: Native support for multiple robots without a central master
3. **Security**: Built-in security features for commercial applications
4. **Quality of Service (QoS)**: Configurable reliability and performance settings
5. **Cross-platform support**: Better Windows and macOS support

## ROS 2 Architecture

### DDS Middleware

ROS 2 uses DDS as its underlying communication middleware. DDS provides:

- **Decentralized discovery**: Nodes find each other automatically
- **Quality of Service (QoS)**: Configurable reliability, durability, and liveliness
- **Language independence**: Support for multiple programming languages
- **Platform independence**: Runs on various operating systems

### Core Components

#### Nodes
- Independent processes that perform computation
- Communicate with other nodes through topics, services, and actions
- Managed by the ROS 2 runtime

#### Topics
- Publish-subscribe communication pattern
- Unidirectional data flow from publishers to subscribers
- Multiple publishers and subscribers can connect to the same topic

#### Services
- Request-response communication pattern
- Synchronous communication
- Single server responds to multiple clients

#### Actions
- Goal-based communication pattern
- Asynchronous with feedback and status updates
- Used for long-running tasks

## Setting Up ROS 2

### Installation

ROS 2 supports multiple distributions. For this course, we'll use ROS Humble Hawksbill (LTS):

```bash
# Ubuntu installation
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update
sudo apt install curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo apt-key add -
sudo sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2.list'
sudo apt update
sudo apt install ros-humble-desktop
sudo apt install python3-colcon-common-extensions
```

### Environment Setup

```bash
# Source the ROS 2 environment
source /opt/ros/humble/setup.bash

# Add to your bashrc to source automatically
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

## Creating Your First ROS 2 Package

### Using colcon

```bash
# Create a workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Create a package
colcon build
source install/setup.bash
```

### Package Structure

```
my_robot_package/
├── CMakeLists.txt
├── package.xml
├── src/
│   └── my_node.cpp
├── include/
│   └── my_robot_package/
│       └── my_header.hpp
├── launch/
│   └── my_launch_file.launch.py
├── config/
│   └── params.yaml
└── test/
    └── test_my_node.cpp
```

## Basic ROS 2 Concepts

### Nodes

A node is an executable that uses ROS 2 to communicate with other nodes. Here's a simple Python node:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalNode()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Topics

Topics enable asynchronous communication between nodes using a publish-subscribe pattern:

```python
# Publisher
publisher = self.create_publisher(String, 'topic_name', 10)
msg = String()
msg.data = 'Hello'
publisher.publish(msg)

# Subscriber
subscriber = self.create_subscription(
    String, 
    'topic_name', 
    self.subscription_callback, 
    10
)

def subscription_callback(self, msg):
    self.get_logger().info('Received: "%s"' % msg.data)
```

### Services

Services provide synchronous request-response communication:

```python
# Service server
self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

def add_two_ints_callback(self, request, response):
    response.sum = request.a + request.b
    self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
    return response

# Service client
client = self.create_client(AddTwoInts, 'add_two_ints')
while not client.wait_for_service(timeout_sec=1.0):
    self.get_logger().info('Service not available, waiting again...')
request = AddTwoInts.Request()
request.a = 41
request.b = 1
future = client.call_async(request)
```

### Actions

Actions are used for long-running tasks with feedback:

```python
# Action server
from rclpy.action import ActionServer
from my_robot_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])
            self.get_logger().info('Feedback: {feedback_msg.sequence}')
            goal_handle.publish_feedback(feedback_msg)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        return result
```

## Quality of Service (QoS) Profiles

QoS profiles allow fine-tuning of communication behavior:

```python
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy

# Create a custom QoS profile
qos_profile = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE
)

# Use the profile when creating publisher/subscriber
publisher = self.create_publisher(String, 'topic', qos_profile)
```

## ROS 2 Tools

### Command Line Tools

```bash
# List all nodes
ros2 node list

# List all topics
ros2 topic list

# Echo messages on a topic
ros2 topic echo /topic_name std_msgs/msg/String

# Call a service
ros2 service call /service_name example_interfaces/srv/AddTwoInts "{a: 1, b: 2}"

# List all actions
ros2 action list

# Get information about a node
ros2 node info /node_name

# Launch a launch file
ros2 launch package_name launch_file.py
```

### rviz2 - Visualization Tool

RViz2 is the 3D visualization tool for ROS 2:

```bash
# Launch RViz2
rviz2

# Add displays for various message types
# Configure visualization parameters
# Save configurations as .rviz files
```

## URDF (Unified Robot Description Format)

URDF describes robot models in XML format:

```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <material name="blue">
    <color rgba="0 0 0.8 1"/>
  </material>

  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.5"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>

  <joint name="base_to_lidar" type="fixed">
    <parent link="base_link"/>
    <child link="lidar_link"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
  </joint>

  <link name="lidar_link">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
    </visual>
  </link>
</robot>
```

## Best Practices

### Node Design
- Keep nodes focused on a single responsibility
- Handle exceptions gracefully
- Use parameters for configuration
- Implement proper cleanup in destructors

### Communication Patterns
- Use topics for streaming data
- Use services for simple request-response
- Use actions for complex, long-running tasks
- Consider QoS settings for your application

### Testing
- Write unit tests for your nodes
- Use launch testing for integration tests
- Test with realistic data and scenarios

## Summary

ROS 2 provides a robust framework for developing complex robotic systems with improved real-time capabilities, security, and multi-robot support. Understanding its architecture and core concepts is essential for building reliable robotic applications.

## Exercises

1. Create a simple ROS 2 package with a publisher and subscriber node.
2. Implement a service server and client for a mathematical operation.
3. Explore the QoS settings and observe their effect on communication.
4. Create a simple URDF model of a wheeled robot.