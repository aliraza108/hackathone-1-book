---
sidebar_position: 4
---

# Week 4: ROS 2 Nodes, Topics, Services, and Actions

## Learning Objectives

By the end of this week, you will be able to:
- Implement ROS 2 nodes with publishers and subscribers
- Create and use custom message types
- Develop service servers and clients
- Implement action servers and clients
- Understand when to use each communication pattern

## ROS 2 Communication Patterns

ROS 2 provides four main communication patterns for nodes to interact:

1. **Topics**: Asynchronous publish/subscribe communication
2. **Services**: Synchronous request/response communication
3. **Actions**: Asynchronous goal-oriented communication with feedback
4. **Parameters**: Configuration values shared between nodes

## Topics: Publish/Subscribe Pattern

Topics enable asynchronous communication between nodes using a publish/subscribe pattern. Publishers send messages to a topic, and subscribers receive messages from the topic.

### Creating a Publisher

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
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
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Creating a Subscriber

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Quality of Service (QoS) Settings

QoS settings allow fine-tuning of communication behavior:

```python
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

# Create a custom QoS profile
qos_profile = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=QoSReliabilityPolicy.BEST_EFFORT
)

# Use the profile when creating publisher/subscriber
publisher = self.create_publisher(String, 'topic', qos_profile)
```

## Services: Request/Response Pattern

Services provide synchronous communication with a request/response pattern.

### Creating a Service Server

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(
            AddTwoInts, 
            'add_two_ints', 
            self.add_two_ints_callback
        )

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Creating a Service Client

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalClientAsync(Node):
    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClientAsync()
    response = minimal_client.send_request(1, 2)
    minimal_client.get_logger().info(
        'Result of add_two_ints: for %d + %d = %d' % 
        (1, 2, response.sum)
    )
    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Actions: Goal-Oriented Communication

Actions are used for long-running tasks that provide feedback and status updates.

### Creating an Action Server

```python
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from my_robot_interfaces.action import Fibonacci  # Custom action definition

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback,
            callback_group=ReentrantCallbackGroup())

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])
            
            self.get_logger().info('Feedback: {feedback_msg.sequence}')
            goal_handle.publish_feedback(feedback_msg)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info('Returning result: {result.sequence}')
        return result

def main(args=None):
    rclpy.init(args=args)
    fibonacci_action_server = FibonacciActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(fibonacci_action_server)
    try:
        executor.spin()
    finally:
        fibonacci_action_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Creating an Action Client

```python
import time
from rclpy.action import ActionClient
from rclpy.node import Node
from my_robot_interfaces.action import Fibonacci  # Custom action definition

class FibonacciActionClient(Node):
    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(
            self,
            Fibonacci,
            'fibonacci')

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)
        
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info('Received feedback: {feedback.sequence}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('Result: {result.sequence}')
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    action_client = FibonacciActionClient()
    action_client.send_goal(10)
    rclpy.spin(action_client)

if __name__ == '__main__':
    main()
```

## Custom Message Types

Custom message types are defined in `.msg` files:

```text
# In msg/Num.msg
int64 num
```

Custom service types in `.srv` files:

```text
# In srv/AddThreeInts.srv
int64 a
int64 b
int64 c
---
int64 sum
```

Custom action types in `.action` files:

```text
# In action/Fibonacci.action
int32 order
---
int32[] sequence
---
int32[] sequence
```

## When to Use Each Pattern

### Topics (Publish/Subscribe)
- Streaming data (sensor readings, robot state)
- Broadcasting information to multiple subscribers
- When you don't need acknowledgment of receipt
- High-frequency data transmission

### Services (Request/Response)
- Simple computations or queries
- When you need guaranteed delivery
- Short-lived operations
- Configuration requests

### Actions (Goal-Oriented)
- Long-running operations (>1 second)
- Operations that provide feedback during execution
- Operations that can be preempted or canceled
- Complex tasks with intermediate results

## Lifecycle Nodes

Lifecycle nodes provide a standardized way to manage node states:

```python
from lifecycle_py.lifecycle_node import LifecycleNode
from lifecycle_py.lifecycle import TransitionCallbackReturn

class LCNode(LifecycleNode):
    def __init__(self, name):
        super().__init__(name)
        self.pub = None
        self.timer = None

    def on_configure(self, state):
        self.pub = self.create_publisher(String, 'lifecycle_chatter', 10)
        self.timer = self.create_timer(1.0, self.pub_cb)
        self.timer.cancel()
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.timer.reset()
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self.timer.cancel()
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        self.destroy_publisher(self.pub)
        self.destroy_timer(self.timer)
        self.pub = None
        self.timer = None
        return TransitionCallbackReturn.SUCCESS

    def pub_cb(self):
        msg = String()
        msg.data = 'Lifecycle node running...'
        self.get_logger().info('Lifecycle node: Publishing {}'.format(msg.data))
        self.pub.publish(msg)
```

## Parameters

Nodes can use parameters for configuration:

```python
import rclpy
from rclpy.node import Node

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')
        
        # Declare parameters with default values
        self.declare_parameter('param_name', 'default_value')
        self.declare_parameter('integer_param', 42)
        self.declare_parameter('double_param', 3.14)
        
        # Get parameter values
        param_value = self.get_parameter('param_name').value
        integer_value = self.get_parameter('integer_param').value
        double_value = self.get_parameter('double_param').value
        
        self.get_logger().info(f'Parameter values: {param_value}, {integer_value}, {double_value}')

def main(args=None):
    rclpy.init(args=args)
    node = ParameterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices

### Topic Design
- Use descriptive topic names
- Consider message frequency and bandwidth
- Use appropriate QoS settings for your application
- Separate time-critical topics from non-critical ones

### Service Design
- Keep services for simple, quick operations
- Use services for configuration and state queries
- Consider using actions for complex operations

### Action Design
- Use actions for operations that take significant time
- Provide meaningful feedback during execution
- Implement proper cancellation handling

### Error Handling
- Always check if services are available before calling
- Handle timeouts appropriately
- Implement retry logic when needed

## Summary

Understanding ROS 2 communication patterns is crucial for building robust robotic systems. Each pattern serves specific purposes and choosing the right one for your use case is essential for system performance and reliability.

## Exercises

1. Create a publisher that publishes sensor data at 10 Hz and a subscriber that processes this data.
2. Implement a service that calculates the distance between two points in 3D space.
3. Design an action that moves a robot to a specified pose with feedback on progress.
4. Create custom message types for your robot's specific needs.