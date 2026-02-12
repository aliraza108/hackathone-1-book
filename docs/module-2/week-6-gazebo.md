---
sidebar_position: 1
---

# Week 6: Robot Simulation with Gazebo

## Learning Objectives

By the end of this week, you will be able to:
- Set up and configure Gazebo for robot simulation
- Create and modify robot models in URDF format
- Implement physics-based simulation with realistic parameters
- Integrate Gazebo with ROS 2 for sensor simulation
- Design custom environments and worlds for robot testing
- Debug and optimize simulation performance

## Introduction to Gazebo

Gazebo is a 3D simulation environment that enables accurate and efficient simulation of robots in complex indoor and outdoor environments. It provides high-fidelity physics simulation, realistic rendering, and convenient programmatic interfaces.

### Key Features of Gazebo
- **Physics Engine**: Based on ODE (Open Dynamics Engine), Bullet, or DART
- **Rendering**: OpenGL-based visualization with realistic lighting
- **Sensors**: Support for various sensor types (cameras, LIDAR, IMU, etc.)
- **Plugins**: Extensible architecture for custom functionality
- **ROS Integration**: Seamless integration with ROS/ROS 2

## Installing and Setting Up Gazebo

### Installation
For ROS 2 Humble:
```bash
sudo apt install ros-humble-gazebo-ros-pkgs
sudo apt install gazebo
```

### Basic Launch
```bash
# Launch Gazebo standalone
gazebo

# Launch with a specific world
gazebo worlds/willowgarage.world
```

## Understanding Gazebo Worlds

### World Files (SDF Format)
Gazebo uses SDF (Simulation Description Format) to define worlds:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="small_room">
    <!-- Include a model from the model database -->
    <include>
      <uri>model://sun</uri>
    </include>
    
    <!-- Define a ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <!-- Define a custom model -->
    <model name="my_robot">
      <pose>0 0 0.5 0 0 0</pose>
      <link name="chassis">
        <pose>0 0 0.1 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>1.0 0.5 0.3</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1.0 0.5 0.3</size>
            </box>
          </geometry>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.083</ixx>
            <iyy>0.146</iyy>
            <izz>0.208</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

## Robot Modeling in URDF

### URDF Basics
URDF (Unified Robot Description Format) describes robot models:

```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <!-- Materials -->
  <material name="blue">
    <color rgba="0 0 0.8 1"/>
  </material>
  <material name="black">
    <color rgba="0 0 0 1"/>
  </material>
  <material name="white">
    <color rgba="1 1 1 1"/>
  </material>

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Wheel Links -->
  <link name="wheel_front_left">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.1"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.1" radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="front_left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_front_left"/>
    <origin xyz="0.3 0.2 0" rpy="1.5707 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
```

### Adding Gazebo-Specific Elements
To make URDF work in Gazebo, add Gazebo-specific elements:

```xml
<!-- Gazebo plugin for ROS control -->
<gazebo>
  <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
    <left_joint>wheel_left_joint</left_joint>
    <right_joint>wheel_right_joint</right_joint>
    <wheel_separation>0.4</wheel_separation>
    <wheel_diameter>0.2</wheel_diameter>
    <command_topic>cmd_vel</command_topic>
    <odometry_topic>odom</odometry_topic>
    <odometry_frame>odom</odometry_frame>
    <robot_base_frame>base_link</robot_base_frame>
  </plugin>
</gazebo>

<!-- Gazebo material definition -->
<gazebo reference="base_link">
  <material>Gazebo/Blue</material>
</gazebo>
```

## Gazebo Plugins

### Sensor Plugins
Gazebo provides various sensor plugins:

```xml
<!-- Camera sensor -->
<gazebo reference="camera_link">
  <sensor type="camera" name="camera1">
    <update_rate>30</update_rate>
    <camera name="head">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>800</width>
        <height>600</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_optical_frame</frame_name>
      <topic_name>image_raw</topic_name>
    </plugin>
  </sensor>
</gazebo>

<!-- LIDAR sensor -->
<gazebo reference="lidar_link">
  <sensor type="ray" name="head_hokuyo_sensor">
    <pose>0 0 0 0 0 0</pose>
    <visualize>false</visualize>
    <update_rate>40</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>
          <resolution>1</resolution>
          <min_angle>-1.570796</min_angle>
          <max_angle>1.570796</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.10</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="gazebo_ros_head_hokuyo_controller" filename="libgazebo_ros_laser.so">
      <topicName>scan</topicName>
      <frameName>lidar_link</frameName>
    </plugin>
  </sensor>
</gazebo>
```

### Controller Plugins
```xml
<!-- Differential drive controller -->
<gazebo>
  <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
    <left_joint>left_wheel_joint</left_joint>
    <right_joint>right_wheel_joint</right_joint>
    <wheel_separation>0.34</wheel_separation>
    <wheel_diameter>0.15</wheel_diameter>
    <max_wheel_torque>20</max_wheel_torque>
    <max_wheel_acceleration>1.0</max_wheel_acceleration>
    <command_topic>cmd_vel</command_topic>
    <odometry_topic>odom</odometry_topic>
    <odometry_frame>odom</odometry_frame>
    <robot_base_frame>base_footprint</robot_base_frame>
    <publish_odom>true</publish_odom>
    <publish_wheel_tf>true</publish_wheel_tf>
    <publish_odom_tf>true</publish_odom_tf>
  </plugin>
</gazebo>
```

## ROS 2 Integration

### Launching Gazebo with ROS 2
Create a launch file to start Gazebo with ROS 2:

```python
# launch/gazebo_simulation.launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={'world': PathJoinSubstitution([
            FindPackageShare('my_robot_description'),
            'worlds',
            'my_world.sdf'
        ])}.items()
    )
    
    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'my_robot'
        ],
        output='screen'
    )
    
    return LaunchDescription([
        gazebo,
        spawn_entity
    ])
```

### Robot State Publisher
```python
# Launch the robot state publisher
robot_state_publisher = Node(
    package='robot_state_publisher',
    executable='robot_state_publisher',
    name='robot_state_publisher',
    output='screen',
    parameters=[{'use_sim_time': True}],
    arguments=[urdf_file_path]
)
```

## Physics Simulation

### Understanding Physics Parameters
Physics parameters affect how objects behave in simulation:

```xml
<!-- World physics -->
<physics type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
  <gravity>0 0 -9.8</gravity>
</physics>
```

### Tuning Physics for Accuracy
- **Max Step Size**: Smaller values increase accuracy but decrease performance
- **Real Time Factor**: Ratio of simulation time to real time
- **Solver Type**: Different solvers have different characteristics

### Collision Detection
```xml
<!-- Fine-tune collision properties -->
<collision>
  <surface>
    <friction>
      <ode>
        <mu>1.0</mu>
        <mu2>1.0</mu2>
      </ode>
    </friction>
    <bounce>
      <restitution_coefficient>0.0</restitution_coefficient>
      <threshold>100000</threshold>
    </bounce>
  </surface>
</collision>
```

## Sensor Simulation

### Camera Simulation
```xml
<gazebo reference="camera_link">
  <sensor type="camera" name="camera_node">
    <update_rate>30</update_rate>
    <camera name="head">
      <horizontal_fov>1.089</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.05</near>
        <far>300</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_link</frame_name>
      <min_depth>0.1</min_depth>
      <max_depth>100.0</max_depth>
      <always_on>true</always_on>
      <update_rate>30</update_rate>
    </plugin>
  </sensor>
</gazebo>
```

### IMU Simulation
```xml
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>true</visualize>
    <topic>__default_topic__</topic>
    <plugin filename="libgazebo_ros_imu.so" name="imu_plugin">
      <topicName>imu</topicName>
      <bodyName>imu_link</bodyName>
      <updateRateHZ>100.0</updateRateHZ>
      <gaussianNoise>0.0</gaussianNoise>
      <xyzOffset>0 0 0</xyzOffset>
      <rpyOffset>0 0 0</rpyOffset>
      <frameName>imu_link</frameName>
    </plugin>
  </sensor>
</gazebo>
```

## Environment Design

### Creating Custom Worlds
Design custom environments for specific testing scenarios:

```xml
<!-- Maze world example -->
<sdf version="1.7">
  <world name="maze_world">
    <!-- Lighting -->
    <include>
      <uri>model://sun</uri>
    </include>
    
    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <!-- Walls forming a maze -->
    <model name="wall_1">
      <pose>0 5 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>10 0.2 1</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>10 0.2 1</size></box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>10</mass>
          <inertia><ixx>1</ixx><iyy>1</iyy><izz>1</izz></inertia>
        </inertial>
      </link>
    </model>
    
    <!-- Add more walls to create the maze -->
  </world>
</sdf>
```

### Object Placement
Strategically place objects to test robot capabilities:

```xml
<!-- Place objects of interest -->
<model name="target_object">
  <pose>5 3 0.2 0 0 0</pose>
  <static>false</static>
  <link name="object_link">
    <collision name="collision">
      <geometry>
        <sphere><radius>0.1</radius></sphere>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <sphere><radius>0.1</radius></sphere>
      </geometry>
      <material>
        <ambient>1 0 0 1</ambient>
        <diffuse>1 0 0 1</diffuse>
      </material>
    </visual>
    <inertial>
      <mass>0.1</mass>
      <inertia><ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz></inertia>
    </inertial>
  </link>
</model>
```

## Performance Optimization

### Reducing Computational Load
- **Simplify Models**: Use simpler collision geometries where possible
- **Reduce Update Rates**: Lower sensor update rates when high frequency isn't needed
- **Disable Unnecessary Features**: Turn off visualization when running headless

### Physics Optimization
```xml
<!-- Optimized physics settings -->
<physics type="ode">
  <max_step_size>0.01</max_step_size>  <!-- Larger step size for performance -->
  <real_time_update_rate>100</real_time_update_rate>  <!-- Lower update rate -->
  <gravity>0 0 -9.8</gravity>
  <ode>
    <solver>
      <type>quick</type>  <!-- Faster solver -->
      <iters>10</iters>   <!-- Fewer iterations -->
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

## Debugging Techniques

### Checking TF Frames
```bash
# View the TF tree
ros2 run tf2_tools view_frames

# Echo TF transforms
ros2 run tf2_ros tf2_echo map base_link
```

### Verifying Sensor Data
```bash
# Check camera feed
ros2 run image_view image_view

# Check LIDAR data
ros2 run rviz2 rviz2
```

### Physics Debugging
Enable contact visualization in Gazebo to debug physics interactions:

```xml
<!-- Enable contact visualization -->
<world>
  <physics type="ode">
    <enable_contact_visualization>true</enable_contact_visualization>
  </physics>
</world>
```

## Best Practices

### Model Design
1. **Start Simple**: Begin with basic shapes and add complexity gradually
2. **Realistic Inertias**: Calculate inertias based on actual dimensions and mass
3. **Appropriate Friction**: Set friction coefficients based on real materials
4. **Collision vs Visual**: Use simpler geometries for collision than for visuals

### Simulation Design
1. **Validation**: Compare simulation results with real robot behavior
2. **Progressive Complexity**: Start with simple environments and add complexity
3. **Consistent Units**: Use consistent units throughout your models
4. **Documentation**: Comment your SDF/URDF files for future maintenance

## Summary

Gazebo provides a powerful platform for robot simulation with realistic physics and sensor modeling. Understanding how to create accurate robot models, design appropriate environments, and integrate with ROS 2 is crucial for effective robotics development. Proper simulation can significantly accelerate development and testing while reducing costs and risks.

## Exercises

1. Create a simple differential drive robot model in URDF with appropriate Gazebo plugins.
2. Design a custom world with obstacles for navigation testing.
3. Implement a sensor suite (camera, LIDAR, IMU) on your robot model.
4. Tune physics parameters to match real-world robot behavior.