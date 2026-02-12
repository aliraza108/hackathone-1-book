---
sidebar_position: 2
---

# Week 7: Unity for High-Fidelity Rendering

## Learning Objectives

By the end of this week, you will be able to:
- Set up Unity for robotics simulation and visualization
- Import and configure robot models in Unity
- Implement physics-based simulation with Unity's physics engine
- Create realistic sensor simulation using Unity's rendering pipeline
- Integrate Unity with ROS 2 for bidirectional communication
- Design immersive environments for robot testing and visualization
- Optimize Unity scenes for real-time performance

## Introduction to Unity for Robotics

Unity is a powerful real-time 3D development platform that has gained significant traction in robotics for high-fidelity visualization and simulation. Unlike traditional robotics simulators, Unity excels in photorealistic rendering, which is essential for training perception systems and creating immersive teleoperation interfaces.

### Why Unity for Robotics?
- **Photorealistic Rendering**: Advanced lighting and materials for realistic sensor simulation
- **Asset Store**: Extensive library of 3D models, environments, and tools
- **Cross-Platform**: Deploy to various platforms including VR/AR
- **Active Community**: Large developer community with robotics-focused resources
- **Real-time Performance**: Optimized for real-time applications

## Setting Up Unity for Robotics

### Installation
1. Download Unity Hub from unity.com
2. Install Unity Editor (2022.3 LTS recommended for robotics projects)
3. Install required packages via Unity Package Manager

### Unity Robotics Hub
Unity provides the Robotics Hub package for robotics-specific functionality:
- ROS# (ROS Sharp) for ROS communication
- ML-Agents for reinforcement learning
- Perception package for synthetic data generation

## Unity Scene Structure for Robotics

### Basic Scene Setup
```csharp
using UnityEngine;

public class RobotSceneManager : MonoBehaviour
{
    [Header("Robot Configuration")]
    public GameObject robotPrefab;
    public Transform spawnPoint;
    
    [Header("Environment Settings")]
    public Light mainLight;
    public Material floorMaterial;
    
    void Start()
    {
        // Spawn robot at designated location
        Instantiate(robotPrefab, spawnPoint.position, spawnPoint.rotation);
        
        // Configure lighting
        SetupLighting();
        
        // Initialize physics
        Physics.autoSimulation = true;
        Physics.defaultSolverIterations = 8;
    }
    
    void SetupLighting()
    {
        RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Skybox;
        RenderSettings.ambientIntensity = 1.0f;
        
        if (mainLight != null)
        {
            mainLight.type = LightType.Directional;
            mainLight.intensity = 1.0f;
            mainLight.shadows = LightShadows.Soft;
        }
    }
}
```

## Robot Modeling in Unity

### Importing Robot Models
Unity supports various 3D model formats:
- **FBX**: Most common format, supports animations and materials
- **OBJ**: Simple geometry format
- **DAE**: Collada format, good for CAD imports

### Robot Component Structure
```csharp
using UnityEngine;

public class RobotController : MonoBehaviour
{
    [Header("Joint Configuration")]
    public ArticulationBody[] joints;
    public float[] jointLimitsMin;
    public float[] jointLimitsMax;
    
    [Header("Actuator Parameters")]
    public float maxForce = 100f;
    public float maxTorque = 50f;
    
    void Start()
    {
        ConfigureJoints();
    }
    
    void ConfigureJoints()
    {
        for (int i = 0; i < joints.Length; i++)
        {
            if (joints[i] != null)
            {
                // Set joint limits
                var drive = joints[i].xDrive;
                drive.lowerLimit = jointLimitsMin[i];
                drive.upperLimit = jointLimitsMax[i];
                drive.forceLimit = maxForce;
                drive.torqueLimit = maxTorque;
                joints[i].xDrive = drive;
                
                // Set joint type
                joints[i].jointType = ArticulationJointType.RevoluteJoint;
            }
        }
    }
    
    public void MoveJoint(int jointIndex, float targetPosition)
    {
        if (jointIndex >= 0 && jointIndex < joints.Length && joints[jointIndex] != null)
        {
            var drive = joints[jointIndex].xDrive;
            drive.target = targetPosition;
            joints[jointIndex].xDrive = drive;
        }
    }
}
```

### Articulation Bodies vs Rigidbody
Unity's ArticulationBody system is designed for articulated rigid body chains:

```csharp
using UnityEngine;

public class ArticulationChainSetup : MonoBehaviour
{
    public ArticulationBody[] chain;
    
    void Start()
    {
        ConfigureArticulationChain();
    }
    
    void ConfigureArticulationChain()
    {
        for (int i = 0; i < chain.Length; i++)
        {
            if (chain[i] != null)
            {
                // Configure joint properties
                chain[i].jointFriction = 0.1f;
                chain[i].linearDamping = 0.05f;
                chain[i].angularDamping = 0.05f;
                
                // Set drive parameters for actuation
                var drive = chain[i].xDrive;
                drive.forceLimit = 100f;
                drive.damping = 10f;
                drive.stiffness = 100f;
                chain[i].xDrive = drive;
            }
        }
    }
}
```

## Physics Simulation in Unity

### Physics Configuration
Unity's physics engine can be tuned for robotics applications:

```csharp
using UnityEngine;

public class PhysicsManager : MonoBehaviour
{
    [Header("Physics Settings")]
    public float fixedDeltaTime = 0.02f;  // 50 Hz physics update
    public int solverIterations = 8;
    public int solverVelocityIterations = 1;
    
    void Awake()
    {
        // Configure physics settings
        Time.fixedDeltaTime = fixedDeltaTime;
        Physics.defaultSolverIterations = solverIterations;
        Physics.defaultSolverVelocityIterations = solverVelocityIterations;
        
        // Set gravity (Earth standard)
        Physics.gravity = new Vector3(0, -9.81f, 0);
    }
    
    void Update()
    {
        // Monitor physics performance
        Debug.Log($"Physics FPS: {1.0f / Time.fixedDeltaTime}");
    }
}
```

### Collision Detection
```csharp
using UnityEngine;

public class CollisionHandler : MonoBehaviour
{
    void OnCollisionEnter(Collision collision)
    {
        Debug.Log($"Collision detected with {collision.gameObject.name}");
        
        // Calculate impact force
        foreach (ContactPoint contact in collision.contacts)
        {
            Debug.DrawRay(contact.point, contact.normal, Color.red, 2.0f);
            Debug.Log($"Impact force: {collision.impulse.magnitude}");
        }
    }
    
    void OnCollisionStay(Collision collision)
    {
        // Continuous contact monitoring
        foreach (ContactPoint contact in collision.contacts)
        {
            Debug.DrawRay(contact.point, contact.normal, Color.green, 0.1f);
        }
    }
    
    void OnCollisionExit(Collision collision)
    {
        Debug.Log($"Collision ended with {collision.gameObject.name}");
    }
}
```

## Sensor Simulation in Unity

### Camera Sensor Simulation
Unity's rendering pipeline can simulate various camera sensors:

```csharp
using UnityEngine;

[RequireComponent(typeof(Camera))]
public class CameraSensor : MonoBehaviour
{
    [Header("Camera Parameters")]
    public float horizontalFOV = 90f;
    public int resolutionWidth = 640;
    public int resolutionHeight = 480;
    public float nearClip = 0.1f;
    public float farClip = 100f;
    
    [Header("Output Settings")]
    public bool outputDepth = false;
    public bool outputSegmentation = false;
    
    private Camera cam;
    private RenderTexture renderTexture;
    private Texture2D outputTexture;
    
    void Start()
    {
        cam = GetComponent<Camera>();
        SetupCamera();
        CreateRenderTexture();
    }
    
    void SetupCamera()
    {
        // Calculate vertical FOV from horizontal FOV
        float aspectRatio = (float)resolutionWidth / resolutionHeight;
        cam.fieldOfView = 2f * Mathf.Atan(Mathf.Tan(horizontalFOV * Mathf.Deg2Rad / 2f) / aspectRatio) * Mathf.Rad2Deg;
        
        cam.nearClipPlane = nearClip;
        cam.farClipPlane = farClip;
    }
    
    void CreateRenderTexture()
    {
        renderTexture = new RenderTexture(resolutionWidth, resolutionHeight, 24);
        renderTexture.Create();
        cam.targetTexture = renderTexture;
        
        outputTexture = new Texture2D(resolutionWidth, resolutionHeight, TextureFormat.RGB24, false);
    }
    
    public Texture2D CaptureImage()
    {
        // Set the camera to render to our texture
        RenderTexture.active = renderTexture;
        cam.Render();
        
        // Read pixels from the render texture
        outputTexture.ReadPixels(new Rect(0, 0, resolutionWidth, resolutionHeight), 0, 0);
        outputTexture.Apply();
        
        // Reset active render texture
        RenderTexture.active = null;
        
        return outputTexture;
    }
    
    void OnDestroy()
    {
        if (renderTexture != null)
            renderTexture.Release();
    }
}
```

### LIDAR Simulation
```csharp
using UnityEngine;
using System.Collections.Generic;

public class LidarSensor : MonoBehaviour
{
    [Header("LIDAR Parameters")]
    public int horizontalResolution = 720;
    public int verticalResolution = 1;
    public float minAngle = -90f;
    public float maxAngle = 90f;
    public float maxRange = 30f;
    public LayerMask detectionLayers = -1;
    
    [Header("Output")]
    public bool visualizeRays = true;
    
    private float[] ranges;
    private Vector3[] directions;
    
    void Start()
    {
        InitializeLidar();
    }
    
    void InitializeLidar()
    {
        int totalBeams = horizontalResolution * verticalResolution;
        ranges = new float[totalBeams];
        directions = new Vector3[totalBeams];
        
        // Precompute ray directions
        int index = 0;
        for (int v = 0; v < verticalResolution; v++)
        {
            for (int h = 0; h < horizontalResolution; h++)
            {
                float hAngle = Mathf.Lerp(minAngle, maxAngle, (float)h / (horizontalResolution - 1));
                float vAngle = 0f; // For 2D LIDAR
                
                if (verticalResolution > 1)
                {
                    vAngle = Mathf.Lerp(-10f, 10f, (float)v / (verticalResolution - 1));
                }
                
                // Convert angles to direction vector
                float hRad = hAngle * Mathf.Deg2Rad;
                float vRad = vAngle * Mathf.Deg2Rad;
                
                Vector3 direction = new Vector3(
                    Mathf.Cos(vRad) * Mathf.Sin(hRad),
                    Mathf.Sin(vRad),
                    Mathf.Cos(vRad) * Mathf.Cos(hRad)
                );
                
                directions[index] = transform.TransformDirection(direction);
                index++;
            }
        }
    }
    
    void Update()
    {
        ScanEnvironment();
    }
    
    void ScanEnvironment()
    {
        for (int i = 0; i < ranges.Length; i++)
        {
            Vector3 origin = transform.position;
            Vector3 direction = directions[i];
            
            RaycastHit hit;
            if (Physics.Raycast(origin, direction, out hit, maxRange, detectionLayers))
            {
                ranges[i] = hit.distance;
                
                if (visualizeRays)
                    Debug.DrawRay(origin, direction * hit.distance, Color.red, 0.1f);
            }
            else
            {
                ranges[i] = maxRange;
                
                if (visualizeRays)
                    Debug.DrawRay(origin, direction * maxRange, Color.green, 0.1f);
            }
        }
    }
    
    public float[] GetRanges()
    {
        return ranges;
    }
    
    public Vector3[] GetDirections()
    {
        return directions;
    }
}
```

### IMU Simulation
```csharp
using UnityEngine;

public class IMUSensor : MonoBehaviour
{
    [Header("IMU Parameters")]
    public float accelerometerNoise = 0.01f;
    public float gyroscopeNoise = 0.01f;
    public float magnetometerNoise = 0.1f;
    
    [Header("Output")]
    public Vector3 linearAcceleration;
    public Vector3 angularVelocity;
    public Vector3 magneticField;
    
    private Rigidbody rb;
    private Vector3 lastVelocity;
    private float lastTime;
    
    void Start()
    {
        rb = GetComponent<Rigidbody>();
        if (rb == null)
        {
            rb = gameObject.AddComponent<Rigidbody>();
            rb.isKinematic = true; // Don't let physics affect this object
        }
        
        lastTime = Time.time;
    }
    
    void FixedUpdate()
    {
        UpdateIMUData();
    }
    
    void UpdateIMUData()
    {
        float deltaTime = Time.time - lastTime;
        lastTime = Time.time;
        
        if (deltaTime <= 0) return;
        
        // Get current velocity
        Vector3 currentVelocity = (transform.position - lastVelocity) / deltaTime;
        
        // Calculate linear acceleration
        linearAcceleration = (currentVelocity - lastVelocity) / deltaTime;
        lastVelocity = currentVelocity;
        
        // Add noise
        linearAcceleration += Random.insideUnitSphere * accelerometerNoise;
        
        // Get angular velocity from rotation
        Quaternion deltaRotation = transform.rotation * Quaternion.Inverse(transform.rotation);
        angularVelocity = new Vector3(
            deltaRotation.eulerAngles.x,
            deltaRotation.eulerAngles.y,
            deltaRotation.eulerAngles.z
        ) / deltaTime;
        
        // Add noise
        angularVelocity += Random.insideUnitSphere * gyroscopeNoise;
        
        // Simulate magnetic field (Earth's magnetic field approximation)
        magneticField = transform.InverseTransformDirection(Vector3.forward * 25f);
        magneticField += Random.insideUnitSphere * magnetometerNoise;
    }
    
    public Vector3 GetLinearAcceleration()
    {
        return linearAcceleration;
    }
    
    public Vector3 GetAngularVelocity()
    {
        return angularVelocity;
    }
    
    public Vector3 GetMagneticField()
    {
        return magneticField;
    }
}
```

## ROS 2 Integration with Unity

### Using ROS# (ROS Sharp)
ROS# is a Unity package that enables communication with ROS:

```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using RosSharp.RosBridgeClient;
using RosSharp.RosBridgeClient.MessageTypes.Sensor;
using RosSharp.RosBridgeClient.MessageTypes.Geometry;

public class UnityRosConnector : MonoBehaviour
{
    [Header("ROS Connection")]
    public string rosBridgeServerUrl = "ws://192.168.1.100:9090";
    
    [Header("Topics")]
    public string cmdVelTopic = "/cmd_vel";
    public string laserScanTopic = "/scan";
    public string imageTopic = "/camera/image_raw";
    
    private RosSocket rosSocket;
    private UnityMainThreadScheduler unityMainThreadScheduler;
    
    void Start()
    {
        ConnectToRos();
    }
    
    void ConnectToRos()
    {
        unityMainThreadScheduler = gameObject.GetComponent<UnityMainThreadScheduler>();
        if (unityMainThreadScheduler == null)
            unityMainThreadScheduler = gameObject.AddComponent<UnityMainThreadScheduler>();
        
        RosBridgeClient.Protocols.WebSocketNetProtocol protocol = 
            new RosBridgeClient.Protocols.WebSocketNetProtocol(rosBridgeServerUrl);
        
        rosSocket = new RosSocket(protocol, unityMainThreadScheduler);
        
        // Subscribe to topics
        rosSocket.Subscribe<LaserScan>(laserScanTopic, ReceiveLaserScan);
        rosSocket.Subscribe<Image>(imageTopic, ReceiveImage);
        
        Debug.Log($"Connected to ROS bridge at {rosBridgeServerUrl}");
    }
    
    void ReceiveLaserScan(LaserScan laserScanMsg)
    {
        // Process laser scan data
        Debug.Log($"Received laser scan with {laserScanMsg.ranges.Count} ranges");
    }
    
    void ReceiveImage(Image imageMsg)
    {
        // Process image data
        Debug.Log($"Received image: {imageMsg.width}x{imageMsg.height}");
    }
    
    public void PublishCmdVel(float linear, float angular)
    {
        Twist twist = new Twist();
        twist.linear = new Vector3(linear, 0, 0);
        twist.angular = new Vector3(0, 0, angular);
        
        rosSocket.Publish(cmdVelTopic, twist);
    }
    
    void OnApplicationQuit()
    {
        if (rosSocket != null)
            rosSocket.Close();
    }
}
```

### Unity-ROS Bridge Architecture
```csharp
using UnityEngine;
using System.Collections.Generic;

public class RobotBridge : MonoBehaviour
{
    [Header("Robot State")]
    public Transform robotBase;
    public List<Transform> jointTransforms;
    
    [Header("Sensors")]
    public CameraSensor cameraSensor;
    public LidarSensor lidarSensor;
    public IMUSensor imuSensor;
    
    private Dictionary<string, object> sensorData;
    
    void Start()
    {
        sensorData = new Dictionary<string, object>();
    }
    
    void Update()
    {
        // Update sensor data
        UpdateSensorData();
        
        // Send data to ROS
        SendToROS();
    }
    
    void UpdateSensorData()
    {
        sensorData["camera"] = cameraSensor.CaptureImage();
        sensorData["lidar_ranges"] = lidarSensor.GetRanges();
        sensorData["imu_accel"] = imuSensor.GetLinearAcceleration();
        sensorData["imu_gyro"] = imuSensor.GetAngularVelocity();
        sensorData["robot_pose"] = robotBase.position;
        sensorData["robot_orientation"] = robotBase.rotation;
    }
    
    void SendToROS()
    {
        // This would interface with the ROS connector
        // Implementation depends on the specific ROS# setup
    }
    
    public Dictionary<string, object> GetSensorData()
    {
        return sensorData;
    }
}
```

## Environment Design

### Creating Realistic Environments
Unity's terrain system and asset store allow for highly detailed environments:

```csharp
using UnityEngine;

public class EnvironmentGenerator : MonoBehaviour
{
    [Header("Terrain Settings")]
    public int terrainWidth = 1000;
    public int terrainLength = 1000;
    public float terrainHeight = 50f;
    
    [Header("Object Spawning")]
    public GameObject[] obstaclePrefabs;
    public int numberOfObstacles = 20;
    
    [Header("Lighting")]
    public Light sunLight;
    public float dayNightCycleSpeed = 1f;
    
    private Terrain terrain;
    private TerrainData terrainData;
    
    void Start()
    {
        GenerateTerrain();
        AddObstacles();
        SetupLighting();
    }
    
    void GenerateTerrain()
    {
        // Create terrain data
        terrainData = new TerrainData();
        terrainData.heightmapResolution = 513;
        terrainData.size = new Vector3(terrainWidth, terrainHeight, terrainLength);
        
        // Set terrain heightmap
        float[,] heights = new float[terrainData.heightmapResolution, terrainData.heightmapResolution];
        for (int i = 0; i < terrainData.heightmapResolution; i++)
        {
            for (int j = 0; j < terrainData.heightmapResolution; j++)
            {
                heights[i, j] = Mathf.PerlinNoise(
                    (float)i / terrainData.heightmapResolution * 10f,
                    (float)j / terrainData.heightmapResolution * 10f
                ) * 0.3f; // Scale height variation
            }
        }
        
        terrainData.SetHeights(0, 0, heights);
        
        // Create terrain game object
        GameObject terrainObject = Terrain.CreateTerrainGameObject(terrainData);
        terrainObject.transform.position = Vector3.zero;
        
        terrain = terrainObject.GetComponent<Terrain>();
    }
    
    void AddObstacles()
    {
        for (int i = 0; i < numberOfObstacles; i++)
        {
            if (obstaclePrefabs.Length > 0)
            {
                int randomIndex = Random.Range(0, obstaclePrefabs.Length);
                GameObject obstacle = Instantiate(obstaclePrefabs[randomIndex]);
                
                // Position randomly on terrain
                float x = Random.Range(0, terrainWidth);
                float z = Random.Range(0, terrainLength);
                float y = terrain.SampleHeight(new Vector3(x, 0, z)) + 1f;
                
                obstacle.transform.position = new Vector3(x, y, z);
            }
        }
    }
    
    void SetupLighting()
    {
        if (sunLight != null)
        {
            StartCoroutine(DayNightCycle());
        }
    }
    
    IEnumerator DayNightCycle()
    {
        while (true)
        {
            // Rotate the sun to simulate day/night cycle
            sunLight.transform.Rotate(Vector3.right, dayNightCycleSpeed * Time.deltaTime);
            yield return null;
        }
    }
}
```

## Performance Optimization

### Rendering Optimization
```csharp
using UnityEngine;

public class RenderingOptimizer : MonoBehaviour
{
    [Header("LOD Settings")]
    public float lodDistance = 50f;
    public int maxRenderedObjects = 100;
    
    [Header("Quality Settings")]
    public bool useOcclusionCulling = true;
    public bool useLODs = true;
    
    private List<Renderer> activeRenderers;
    
    void Start()
    {
        ConfigureQualitySettings();
        activeRenderers = new List<Renderer>();
    }
    
    void ConfigureQualitySettings()
    {
        QualitySettings.vSyncCount = 0; // Disable V-Sync for consistent frame rate
        Application.targetFrameRate = 60; // Target 60 FPS
        
        // Reduce shadow resolution for performance
        QualitySettings.shadowResolution = ShadowResolution.Medium;
        QualitySettings.shadowDistance = 50f;
    }
    
    void Update()
    {
        if (useLODs)
            ManageLODs();
        
        if (useOcclusionCulling)
            PerformOcclusionCulling();
    }
    
    void ManageLODs()
    {
        Renderer[] allRenderers = FindObjectsOfType<Renderer>();
        
        foreach (Renderer renderer in allRenderers)
        {
            float distance = Vector3.Distance(renderer.transform.position, Camera.main.transform.position);
            
            if (distance > lodDistance)
            {
                // Simplify or hide distant objects
                renderer.enabled = false;
            }
            else
            {
                renderer.enabled = true;
            }
        }
    }
    
    void PerformOcclusionCulling()
    {
        // Unity's built-in occlusion culling system handles this automatically
        // when enabled in the scene
    }
}
```

### Physics Optimization
```csharp
using UnityEngine;

public class PhysicsOptimizer : MonoBehaviour
{
    [Header("Physics Settings")]
    public float fixedDeltaTime = 0.02f; // 50 Hz
    public int solverIterations = 6;
    public float sleepThreshold = 0.005f;
    
    void Start()
    {
        ConfigurePhysics();
    }
    
    void ConfigurePhysics()
    {
        Time.fixedDeltaTime = fixedDeltaTime;
        Physics.defaultSolverIterations = solverIterations;
        Physics.sleepThreshold = sleepThreshold;
        
        // Optimize collision matrix
        Physics.IgnoreLayerCollision(0, 0, true); // Ignore collisions between default layers if not needed
    }
    
    public void OptimizeRigidbody(Rigidbody rb)
    {
        // Optimize individual rigidbody settings
        rb.maxAngularVelocity = 50f;
        rb.sleepThreshold = sleepThreshold;
        rb.interpolation = RigidbodyInterpolation.Interpolate;
    }
}
```

## Best Practices

### Model Preparation
1. **Polygon Count**: Keep polygon counts reasonable for real-time performance
2. **Texture Resolution**: Use appropriate texture sizes (2048x2048 max for most applications)
3. **UV Mapping**: Ensure proper UV mapping for textures
4. **Center of Mass**: Set accurate center of mass for physics simulation

### Scene Organization
1. **Layer Management**: Use Unity layers for efficient collision detection
2. **Tagging**: Tag objects appropriately for easy identification
3. **Prefab Usage**: Use prefabs for reusable robot and environment components
4. **Scene Structure**: Organize scenes hierarchically for easier management

### Performance Considerations
1. **Fixed Update**: Use FixedUpdate for physics-related updates
2. **Object Pooling**: Implement object pooling for frequently instantiated objects
3. **Coroutines**: Use coroutines for time-based operations
4. **Profiler**: Regularly use Unity profiler to identify bottlenecks

## Summary

Unity provides a powerful platform for high-fidelity robotics simulation and visualization. Its advanced rendering capabilities, physics engine, and extensibility make it ideal for creating realistic environments and sensor simulations. When combined with ROS integration, Unity becomes a valuable tool for robotics development, testing, and visualization.

## Exercises

1. Create a simple robot model in Unity with articulated joints.
2. Implement a camera sensor simulation with adjustable parameters.
3. Design a complex environment with varied terrain and obstacles.
4. Integrate your Unity simulation with ROS for bidirectional communication.