---
sidebar_position: 5
---

# Week 5: ROS 2 Python Integration (rclpy)

## Learning Objectives

By the end of this week, you will be able to:
- Use rclpy to create ROS 2 nodes in Python
- Implement publishers, subscribers, services, and actions using rclpy
- Understand the differences between rclpy and rospy
- Create custom message and service definitions
- Integrate Python libraries with ROS 2 nodes
- Debug and profile ROS 2 Python nodes

## Introduction to rclpy

rclpy is the Python client library for ROS 2, providing Python bindings for the ROS 2 client library (rcl). It allows Python developers to create ROS 2 nodes and interact with the ROS 2 ecosystem.

### Key Differences from rospy

1. **Architecture**: rclpy is built on the new ROS 2 architecture with DDS as the middleware
2. **Threading**: Improved threading model compared to rospy
3. **Multi-node processes**: Better support for multiple nodes in a single process
4. **Parameters**: Enhanced parameter system with type safety
5. **Quality of Service**: Configurable QoS settings for different communication needs

## Basic Node Structure

### Minimal Node Example

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_node')
        self.publisher = self.create_publisher(String, 'topic', 10)
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.counter = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.counter}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.counter += 1

def main(args=None):
    rclpy.init(args=args)
    node = MinimalNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Publishers and Subscribers

### Publisher with Custom QoS

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from std_msgs.msg import String

class PublisherNode(Node):
    def __init__(self):
        super().__init__('publisher_node')
        
        # Create a custom QoS profile
        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        
        self.publisher = self.create_publisher(String, 'chatter', qos_profile)
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    node = PublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Subscriber with Message Filters

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from message_filters import ApproximateTimeSynchronizer, Subscriber

class SynchronizedSubscriber(Node):
    def __init__(self):
        super().__init__('sync_subscriber')
        
        # Create subscribers
        self.string_sub = Subscriber(self, String, 'string_topic')
        self.laser_sub = Subscriber(self, LaserScan, 'laser_scan')
        
        # Synchronize messages based on timestamps
        ats = ApproximateTimeSynchronizer(
            [self.string_sub, self.laser_sub], 
            queue_size=10, 
            slop=0.1
        )
        ats.registerCallback(self.sync_callback)

    def sync_callback(self, string_msg, laser_msg):
        self.get_logger().info(
            f'Received synchronized messages: {string_msg.data}, '
            f'Laser ranges: {len(laser_msg.ranges)}'
        )

def main(args=None):
    rclpy.init(args=args)
    node = SynchronizedSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Services and Clients

### Service Server with Error Handling

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class ServiceServer(Node):
    def __init__(self):
        super().__init__('service_server')
        self.srv = self.create_service(
            AddTwoInts, 
            'add_two_ints', 
            self.add_two_ints_callback
        )

    def add_two_ints_callback(self, request, response):
        try:
            # Perform the calculation
            result = request.a + request.b
            
            # Validate the result
            if result > 1000000:
                self.get_logger().warn('Large result detected')
            
            response.sum = result
            self.get_logger().info(f'Request: {request.a} + {request.b} = {response.sum}')
            
        except Exception as e:
            self.get_logger().error(f'Service error: {e}')
            response.sum = 0  # Return a safe default
            
        return response

def main(args=None):
    rclpy.init(args=args)
    node = ServiceServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Service Client with Timeout

```python
import rclpy
from rclpy.node import Node
from rclpy.task import Future
from example_interfaces.srv import AddTwoInts
import time

class ServiceClient(Node):
    def __init__(self):
        super().__init__('service_client')
        self.client = self.create_client(AddTwoInts, 'add_two_ints')
        
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        
        self.request = AddTwoInts.Request()

    def send_request(self, a, b, timeout=5.0):
        self.request.a = a
        self.request.b = b
        
        future = self.client.call_async(self.request)
        
        # Wait for the result with timeout
        start_time = time.time()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            
            if future.done():
                return future.result()
                
            if time.time() - start_time > timeout:
                self.get_logger().error('Service call timed out')
                return None
                
        return None

def main(args=None):
    rclpy.init(args=args)
    client = ServiceClient()
    
    response = client.send_request(10, 20)
    if response is not None:
        client.get_logger().info(f'Result: {response.sum}')
    else:
        client.get_logger().error('Failed to get response')
    
    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Actions

### Action Server with Complex Logic

```python
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from my_robot_interfaces.action import NavigateToPose  # Custom action
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose as Nav2NavigateToPose

class NavigateToPoseActionServer(Node):
    def __init__(self):
        super().__init__('navigate_to_pose_action_server')
        self._goal_handle = None
        self._action_server = ActionServer(
            self,
            NavigateToPose,
            'navigate_to_pose',
            execute_callback=self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            handle_accepted_callback=self.handle_accepted_callback,
            cancel_callback=self.cancel_callback
        )

    def destroy(self):
        super().destroy()
        self._action_server.destroy()

    def goal_callback(self, goal_request):
        """Accept or reject a client request to begin an action."""
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def handle_accepted_callback(self, goal_handle):
        """Handle an accepted goal."""
        self.get_logger().info('Goal accepted')
        self._goal_handle = goal_handle
        goal_handle.execute()

    def cancel_callback(self, goal):
        """Accept or reject a client request to cancel an action."""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute the goal."""
        self.get_logger().info('Executing goal...')
        
        # Get the goal
        target_pose = goal_handle.request.pose
        
        # Feedback and result
        feedback_msg = NavigateToPose.Feedback()
        result_msg = NavigateToPose.Result()
        
        # Simulate navigation
        for i in range(1, 101):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                result_msg.success = False
                return result_msg
            
            # Update feedback
            feedback_msg.distance_remaining = 100 - i
            goal_handle.publish_feedback(feedback_msg)
            
            # Sleep to simulate work
            await rclpy.sleep(0.1)
        
        # Complete the goal
        goal_handle.succeed()
        result_msg.success = True
        result_msg.message = "Navigation completed successfully"
        
        self.get_logger().info('Returning result: %s' % result_msg.success)
        return result_msg

def main(args=None):
    rclpy.init(args=args)
    action_server = NavigateToPoseActionServer()
    
    executor = MultiThreadedExecutor()
    executor.add_node(action_server)
    
    try:
        executor.spin()
    finally:
        action_server.destroy()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Integrating Python Libraries

### Using NumPy with ROS 2 Messages

```python
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import struct

class PointCloudNode(Node):
    def __init__(self):
        super().__init__('pointcloud_node')
        self.publisher = self.create_publisher(PointCloud2, 'pointcloud', 10)
        self.timer = self.create_timer(1.0, self.publish_pointcloud)
        
    def create_pointcloud_msg(self, points):
        """Create a PointCloud2 message from a numpy array of points."""
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'sensor_frame'
        
        # Define the fields
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        # Pack the data
        data = []
        for point in points:
            data.extend(struct.pack('fff', point[0], point[1], point[2]))
        
        # Create the message
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(points)
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 12  # 3 floats * 4 bytes each
        msg.row_step = msg.point_step * msg.width
        msg.data = b''.join(data)
        msg.is_dense = True
        
        return msg

    def publish_pointcloud(self):
        # Generate random points using NumPy
        points = np.random.rand(100, 3) * 10  # 100 random points in a 10x10x10 cube
        
        # Create and publish the message
        msg = self.create_pointcloud_msg(points)
        self.publisher.publish(msg)
        self.get_logger().info(f'Published pointcloud with {len(points)} points')

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Using OpenCV for Image Processing

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageProcessorNode(Node):
    def __init__(self):
        super().__init__('image_processor')
        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, 'processed_image', 10)
        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Process the image (example: apply Gaussian blur)
            processed_image = cv2.GaussianBlur(cv_image, (15, 15), 0)
            
            # Convert back to ROS Image message
            processed_msg = self.bridge.cv2_to_imgmsg(processed_image, "bgr8")
            processed_msg.header = msg.header  # Preserve header
            
            # Publish the processed image
            self.image_pub.publish(processed_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Parameters and Configuration

### Dynamic Parameters

```python
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor

class DynamicParamNode(Node):
    def __init__(self):
        super().__init__('dynamic_param_node')
        
        # Declare parameters with descriptions
        self.declare_parameter(
            'threshold', 
            0.5, 
            ParameterDescriptor(description='Detection threshold')
        )
        
        self.declare_parameter(
            'algorithm', 
            'sift', 
            ParameterDescriptor(description='Feature detection algorithm')
        )
        
        # Register parameter callback
        self.add_on_set_parameters_callback(self.param_callback)
        
        # Timer to periodically use parameters
        self.timer = self.create_timer(1.0, self.use_params)
        
    def param_callback(self, params):
        """Callback for parameter changes."""
        for param in params:
            if param.name == 'threshold' and param.type_ == Parameter.Type.DOUBLE:
                if param.value < 0.0 or param.value > 1.0:
                    return SetParametersResult(successful=False, reason='Threshold must be between 0 and 1')
        
        return SetParametersResult(successful=True)
    
    def use_params(self):
        """Use the current parameter values."""
        threshold = self.get_parameter('threshold').value
        algorithm = self.get_parameter('algorithm').value
        
        self.get_logger().info(f'Using threshold: {threshold}, algorithm: {algorithm}')

def main(args=None):
    rclpy.init(args=args)
    node = DynamicParamNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Debugging and Profiling

### Logging Best Practices

```python
import rclpy
from rclpy.node import Node
import traceback

class LoggingExampleNode(Node):
    def __init__(self):
        super().__init__('logging_example')
        
        # Set up different log levels
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)
        
        # Log different types of messages
        self.get_logger().debug('Debug information')
        self.get_logger().info('General information')
        self.get_logger().warn('Warning message')
        self.get_logger().error('Error message')
        self.get_logger().fatal('Fatal error message')
        
        # Example of exception logging
        try:
            result = 10 / 0
        except Exception as e:
            self.get_logger().error(f'Exception occurred: {e}')
            self.get_logger().error(traceback.format_exc())
        
        self.timer = self.create_timer(2.0, self.periodic_log)
    
    def periodic_log(self):
        self.get_logger().info('Periodic log message')

def main(args=None):
    rclpy.init(args=args)
    node = LoggingExampleNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Performance Monitoring

```python
import rclpy
from rclpy.node import Node
import time
from std_msgs.msg import Float64

class PerformanceMonitorNode(Node):
    def __init__(self):
        super().__init__('performance_monitor')
        
        self.latency_pub = self.create_publisher(Float64, 'callback_latency', 10)
        self.frequency_pub = self.create_publisher(Float64, 'callback_frequency', 10)
        
        # Track timing
        self.last_call_time = time.time()
        self.callback_times = []
        
        # Create a timer with a callback
        self.timer = self.create_timer(0.1, self.timed_callback)
    
    def timed_callback(self):
        start_time = time.time()
        
        # Simulate work
        time.sleep(0.01)  # Simulate 10ms of work
        
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to ms
        
        # Calculate actual frequency
        current_time = time.time()
        period = current_time - self.last_call_time
        frequency = 1.0 / period if period > 0 else 0
        self.last_call_time = current_time
        
        # Store timing data
        self.callback_times.append(latency)
        if len(self.callback_times) > 100:
            self.callback_times.pop(0)
        
        # Publish metrics
        latency_msg = Float64()
        latency_msg.data = latency
        self.latency_pub.publish(latency_msg)
        
        freq_msg = Float64()
        freq_msg.data = frequency
        self.frequency_pub.publish(freq_msg)
        
        # Log statistics periodically
        if len(self.callback_times) % 10 == 0:
            avg_latency = sum(self.callback_times) / len(self.callback_times)
            self.get_logger().info(f'Avg latency: {avg_latency:.2f}ms, Frequency: {frequency:.2f}Hz')

def main(args=None):
    rclpy.init(args=args)
    node = PerformanceMonitorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Testing with rclpy

### Unit Testing

```python
import unittest
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from example_interfaces.srv import AddTwoInts

class TestNode(Node):
    def __init__(self):
        super().__init__('test_node')
        self.publisher = self.create_publisher(String, 'test_topic', 10)
        self.service = self.create_service(
            AddTwoInts, 
            'test_add_service', 
            self.add_callback
        )
        self.received_messages = []

    def add_callback(self, request, response):
        response.sum = request.a + request.b
        return response

class TestRclpy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = TestNode()

    def tearDown(self):
        self.node.destroy_node()

    def test_publisher_exists(self):
        """Test that publisher is created."""
        self.assertIsNotNone(self.node.publisher)

    def test_service_callback(self):
        """Test service callback functionality."""
        request = AddTwoInts.Request()
        request.a = 5
        request.b = 3
        
        response = self.node.add_callback(request, AddTwoInts.Response())
        self.assertEqual(response.sum, 8)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
```

## Best Practices

### Node Organization

1. **Modular Design**: Break complex nodes into smaller, focused nodes
2. **Resource Management**: Always clean up resources in destroy_node()
3. **Error Handling**: Implement proper exception handling
4. **Logging**: Use appropriate log levels for different types of messages
5. **Configuration**: Use parameters for configurable values

### Performance Considerations

1. **Threading**: Use ReentrantCallbackGroup when callbacks need to run concurrently
2. **Memory**: Be mindful of memory usage, especially with large messages
3. **Timing**: Use appropriate timer periods for different tasks
4. **QoS**: Select appropriate QoS settings for your use case

## Summary

rclpy provides a powerful interface for developing ROS 2 nodes in Python. Understanding its capabilities and best practices is essential for building robust robotic applications. The integration with popular Python libraries like NumPy and OpenCV enables sophisticated robotics applications.

## Exercises

1. Create a node that subscribes to sensor data, processes it using NumPy, and publishes the results.
2. Implement a service that performs a complex calculation using a Python library.
3. Build an action server that controls a robot's movement with feedback.
4. Create a parameterized node that adjusts its behavior based on runtime parameters.