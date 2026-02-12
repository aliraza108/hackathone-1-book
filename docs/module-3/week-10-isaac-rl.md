---
sidebar_position: 3
---

# Week 10: Reinforcement Learning and Sim-to-Real Transfer

## Learning Objectives

By the end of this week, you will be able to:
- Understand reinforcement learning fundamentals for robotics
- Implement RL algorithms for robotic control tasks
- Design simulation environments for RL training
- Apply domain randomization techniques for sim-to-real transfer
- Evaluate and validate RL policies in simulation and reality
- Understand the challenges and solutions in sim-to-real transfer
- Implement curriculum learning for complex robotic tasks

## Reinforcement Learning in Robotics

### Overview of RL for Robotics
Reinforcement Learning (RL) is a powerful approach for learning robotic behaviors through trial and error. In robotics, RL can be used for:

- **Motor Control**: Learning complex movement patterns
- **Manipulation**: Acquiring dexterous manipulation skills
- **Navigation**: Learning to navigate complex environments
- **Task Planning**: Sequencing actions to achieve goals

### Key RL Concepts in Robotics
- **State Space**: Robot configuration, sensor readings, environment state
- **Action Space**: Motor commands, joint torques, end-effector velocities
- **Reward Function**: Encourages desired behaviors and penalizes undesired ones
- **Policy**: Maps states to actions (the learned behavior)
- **Value Function**: Estimates expected future rewards

### RL Algorithm Taxonomy
```cpp
// Example of a simple RL policy in Isaac
#include "engine/alice/alice.hpp"
#include "engine/gems/reinforcement_learning/dqn_agent.hpp"

namespace isaac {
namespace rl {

class RobotDQNPolicy : public Codelet {
 public:
  void start() override {
    // Initialize DQN agent
    dqn_agent_ = std::make_unique<DQNAgent>();
    dqn_agent_->initialize(state_dim_, action_dim_, learning_rate_);
    
    // Set up reward shaping
    setupRewardFunction();
    
    // Start training if in simulation, otherwise load trained model
    if (is_training_mode_) {
      startTraining();
    } else {
      loadTrainedModel();
    }
    
    tickPeriodically(std::chrono::milliseconds(control_freq_ms_));
  }

  void tick() override {
    // Get current state from sensors
    auto state = getCurrentState();
    
    // Get action from policy
    int action;
    if (is_training_mode_) {
      action = dqn_agent_->getAction(state, epsilon_);
    } else {
      action = dqn_agent_->getBestAction(state);
    }
    
    // Execute action
    executeAction(action);
    
    // In training mode, collect experience and update policy
    if (is_training_mode_) {
      auto next_state = getCurrentState();  // After action execution
      float reward = computeReward(state, action, next_state);
      bool done = checkEpisodeDone(next_state);
      
      // Store experience
      dqn_agent_->remember(state, action, reward, next_state, done);
      
      // Train on batch of experiences
      if (step_count_ % train_freq_ == 0) {
        dqn_agent_->replay(batch_size_);
      }
      
      // Update target network periodically
      if (step_count_ % target_update_freq_ == 0) {
        dqn_agent_->updateTargetNetwork();
      }
      
      // Decay exploration rate
      epsilon_ = std::max(epsilon_min_, epsilon_ * epsilon_decay_);
    }
    
    step_count_++;
  }

 private:
  void setupRewardFunction() {
    // Define reward components
    reward_weights_["progress"] = 1.0f;
    reward_weights_["energy_efficiency"] = 0.1f;
    reward_weights_["safety"] = 2.0f;
    reward_weights_["task_completion"] = 10.0f;
  }
  
  std::vector<float> getCurrentState() {
    // Combine multiple sensor modalities
    std::vector<float> state;
    
    // Add joint positions and velocities
    auto joint_positions = getJointPositions();
    auto joint_velocities = getJointVelocities();
    state.insert(state.end(), joint_positions.begin(), joint_positions.end());
    state.insert(state.end(), joint_velocities.begin(), joint_velocities.end());
    
    // Add IMU data
    auto imu_data = getIMUData();
    state.insert(state.end(), imu_data.begin(), imu_data.end());
    
    // Add camera data (processed features, not raw pixels)
    auto visual_features = getVisualFeatures();
    state.insert(state.end(), visual_features.begin(), visual_features.end());
    
    // Add task-specific state (e.g., object positions)
    auto task_state = getTaskState();
    state.insert(state.end(), task_state.begin(), task_state.end());
    
    return state;
  }
  
  float computeReward(const std::vector<float>& state, 
                     int action, 
                     const std::vector<float>& next_state) {
    float total_reward = 0.0f;
    
    // Progress toward goal
    float progress_reward = computeProgressReward(state, next_state);
    total_reward += reward_weights_["progress"] * progress_reward;
    
    // Energy efficiency (penalize excessive actuator effort)
    float energy_penalty = computeEnergyPenalty(action);
    total_reward -= reward_weights_["energy_efficiency"] * energy_penalty;
    
    // Safety (penalize dangerous states)
    float safety_penalty = computeSafetyPenalty(next_state);
    total_reward -= reward_weights_["safety"] * safety_penalty;
    
    // Task completion bonus
    if (checkTaskCompleted(next_state)) {
      total_reward += reward_weights_["task_completion"];
    }
    
    return total_reward;
  }
  
  void executeAction(int action) {
    // Convert discrete action to continuous motor commands
    auto motor_commands = convertDiscreteToContinuousAction(action);
    
    // Send commands to robot
    sendMotorCommands(motor_commands);
  }

  std::unique_ptr<DQNAgent> dqn_agent_;
  int state_dim_ = 128;  // Example dimension
  int action_dim_ = 16;  // Example dimension
  float learning_rate_ = 0.001f;
  bool is_training_mode_ = true;
  int control_freq_ms_ = 50;  // 20 Hz control frequency
  float epsilon_ = 1.0f;
  float epsilon_min_ = 0.01f;
  float epsilon_decay_ = 0.995f;
  int train_freq_ = 4;
  int target_update_freq_ = 1000;
  int batch_size_ = 32;
  int step_count_ = 0;
  
  std::map<std::string, float> reward_weights_;
};

}  // namespace rl
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::rl::RobotDQNPolicy);
```

## Simulation Environments for RL

### Designing Effective Simulation Environments
Creating simulation environments that facilitate effective RL training:

```cpp
#include "engine/alice/alice.hpp"
#include "engine/gems/simulation/physics_engine.hpp"

namespace isaac {
namespace simulation {

class RLTrainingEnvironment : public Codelet {
 public:
  void start() override {
    // Initialize physics engine
    physics_engine_ = std::make_unique<PhysicsEngine>();
    physics_engine_->initialize();
    
    // Set up simulation parameters
    setupSimulationParameters();
    
    // Create randomized environments
    createRandomizedScenes();
    
    tickPeriodically(std::chrono::microseconds(1000));  // High-frequency physics
  }

  void tick() override {
    // Update physics simulation
    physics_engine_->stepSimulation(time_step_);
    
    // Update robot sensors
    updateRobotSensors();
    
    // Check for collisions and other events
    checkSimulationEvents();
    
    // If in training mode, occasionally reset environment
    if (is_training_mode_ && shouldResetEnvironment()) {
      resetEnvironment();
    }
  }

 private:
  void setupSimulationParameters() {
    // Physics parameters
    physics_engine_->setGravity({0, 0, -9.81f});
    physics_engine_->setTimeStep(time_step_);
    
    // Domain randomization parameters
    randomization_params_["floor_friction"] = {0.4f, 0.8f};
    randomization_params_["object_mass"] = {0.1f, 2.0f};
    randomization_params_["lighting_condition"] = {0.5f, 2.0f};
    randomization_params_["texture_variation"] = {0.0f, 1.0f};
  }
  
  void createRandomizedScenes() {
    // Create multiple scene variations
    for (int i = 0; i < num_scenes_; i++) {
      auto scene = createBaseScene();
      
      // Randomize physical properties
      randomizePhysicalProperties(scene);
      
      // Randomize visual appearance
      randomizeVisualProperties(scene);
      
      // Randomize object placements
      randomizeObjectPlacements(scene);
      
      scenes_.push_back(scene);
    }
  }
  
  void randomizePhysicalProperties(Scene& scene) {
    // Randomize friction coefficients
    float floor_friction = uniformRandom(
        randomization_params_["floor_friction"].first,
        randomization_params_["floor_friction"].second);
    scene.setFloorFriction(floor_friction);
    
    // Randomize object masses
    for (auto& obj : scene.getObjects()) {
      float mass = uniformRandom(
          randomization_params_["object_mass"].first,
          randomization_params_["object_mass"].second);
      obj.setMass(mass);
    }
  }
  
  void randomizeVisualProperties(Scene& scene) {
    // Randomize lighting
    float lighting_strength = uniformRandom(
        randomization_params_["lighting_condition"].first,
        randomization_params_["lighting_condition"].second);
    scene.setLightingStrength(lighting_strength);
    
    // Randomize textures
    if (uniformRandom(0.0f, 1.0f) < randomization_params_["texture_variation"].second) {
      scene.applyRandomTextures();
    }
  }
  
  void resetEnvironment() {
    // Select a random scene
    int random_scene_idx = uniformRandomInt(0, scenes_.size() - 1);
    current_scene_ = scenes_[random_scene_idx];
    
    // Randomize robot starting position
    auto random_start_pos = getRandomStartPosition();
    robot_->setPosition(random_start_pos);
    
    // Randomize task-relevant objects
    randomizeTaskObjects();
    
    // Reset episode counter
    episode_step_ = 0;
  }

  std::unique_ptr<PhysicsEngine> physics_engine_;
  float time_step_ = 1.0f / 1000.0f;  // 1000 Hz physics update
  bool is_training_mode_ = true;
  int num_scenes_ = 50;
  std::vector<Scene> scenes_;
  Scene current_scene_;
  std::map<std::string, std::pair<float, float>> randomization_params_;
  int episode_step_ = 0;
  int max_episode_steps_ = 1000;
};

}  // namespace simulation
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::simulation::RLTrainingEnvironment);
```

## Domain Randomization

### Understanding Domain Randomization
Domain randomization is a technique to improve sim-to-real transfer by randomizing simulation parameters:

```cpp
#include "engine/alice/alice.hpp"
#include "engine/gems/randomization/domain_randomizer.hpp"

namespace isaac {
namespace randomization {

class DomainRandomizer : public Codelet {
 public:
  void start() override {
    // Initialize randomization parameters
    initializeRandomizationParameters();
    
    // Set up parameter distributions
    setupParameterDistributions();
    
    // Apply initial randomization
    applyRandomization();
    
    tickOnMessage(rx_reset_signal());
  }

  void tick() override {
    // Apply randomization when reset signal received
    if (rx_reset_signal().available()) {
      auto reset_msg = rx_reset_signal().popLatest();
      if (reset_msg && reset_msg.value().trigger_randomization()) {
        applyRandomization();
      }
    }
  }

 private:
  void initializeRandomizationParameters() {
    // Physical parameters
    randomization_config_["mass_variance"] = {0.8f, 1.2f};  // ±20% variance
    randomization_config_["friction_variance"] = {0.5f, 1.5f};
    randomization_config_["com_offset"] = {-0.01f, 0.01f};  // ±1cm offset
    
    // Visual parameters
    randomization_config_["light_intensity"] = {0.7f, 1.3f};
    randomization_config_["camera_noise"] = {0.0f, 0.05f};
    randomization_config_["texture_randomization"] = {0.0f, 1.0f};
    
    // Dynamics parameters
    randomization_config_["motor_torque_limits"] = {0.9f, 1.1f};
    randomization_config_["sensor_bias"] = {-0.01f, 0.01f};
    randomization_config_["delay_variance"] = {0.0f, 0.02f};  // ±20ms delay
  }
  
  void setupParameterDistributions() {
    // Set up different types of distributions
    distributions_["mass"] = std::make_unique<UniformDistribution>(
        randomization_config_["mass_variance"].first,
        randomization_config_["mass_variance"].second);
        
    distributions_["friction"] = std::make_unique<NormalDistribution>(
        1.0f,  // mean
        0.1f   // std dev
    );
    
    distributions_["lighting"] = std::make_unique<BetaDistribution>(
        2.0f,  // alpha
        2.0f   // beta
    );
  }
  
  void applyRandomization() {
    // Apply randomization to physical properties
    applyMassRandomization();
    applyFrictionRandomization();
    applyCOMRandomization();
    
    // Apply randomization to visual properties
    applyLightingRandomization();
    applyCameraRandomization();
    
    // Apply randomization to dynamics
    applyMotorRandomization();
    applySensorRandomization();
    
    LOG_INFO("Applied domain randomization");
  }
  
  void applyMassRandomization() {
    auto& robot = getRobotModel();
    for (auto& link : robot.getLinks()) {
      float original_mass = link.getOriginalMass();
      float random_factor = distributions_["mass"]->sample();
      link.setMass(original_mass * random_factor);
    }
  }
  
  void applyFrictionRandomization() {
    auto& robot = getRobotModel();
    for (auto& joint : robot.getJoints()) {
      float original_friction = joint.getOriginalFriction();
      float random_factor = distributions_["friction"]->sample();
      joint.setFriction(original_friction * random_factor);
    }
  }
  
  void applyLightingRandomization() {
    auto& renderer = getRenderer();
    float intensity_factor = distributions_["lighting"]->sample();
    renderer.setLightIntensityFactor(intensity_factor);
  }

  std::map<std::string, std::pair<float, float>> randomization_config_;
  std::map<std::string, std::unique_ptr<Distribution>> distributions_;
  
  // Communication channels
  ISAAC_PROTO_RX(ResetSignalProto, reset_signal);
};

}  // namespace randomization
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::randomization::DomainRandomizer);
```

## Sim-to-Real Transfer Techniques

### System Identification and Modeling
Understanding the differences between simulation and reality:

```cpp
#include "engine/alice/alice.hpp"
#include "engine/gems/system_identification/parameter_estimator.hpp"

namespace isaac {
namespace sysid {

class SimRealAdapter : public Codelet {
 public:
  void start() override {
    // Initialize parameter estimator
    param_estimator_ = std::make_unique<ParameterEstimator>();
    
    // Load simulation model
    loadSimulationModel();
    
    // Start with simulation parameters
    current_params_ = simulation_params_;
    
    // If in real robot mode, start adaptation
    if (is_real_robot_) {
      startAdaptation();
    }
    
    tickPeriodically(std::chrono::milliseconds(10));  // 100 Hz adaptation
  }

  void tick() override {
    if (is_real_robot_) {
      // Collect data from real robot
      auto real_data = collectRealData();
      
      // Compare with simulation predictions
      auto sim_prediction = predictWithSimulation(real_data.input);
      
      // Update model parameters based on discrepancy
      updateModelParameters(real_data, sim_prediction);
      
      // Publish adapted parameters
      publishAdaptedParameters();
    }
  }

 private:
  void loadSimulationModel() {
    // Load physics and dynamics model from URDF/SDF
    simulation_model_ = loadModelFromFile(sim_model_path_);
    
    // Extract nominal parameters
    simulation_params_ = simulation_model_.getParameters();
    current_params_ = simulation_params_;
  }
  
  void startAdaptation() {
    // Initialize adaptation algorithm
    adaptation_algorithm_ = std::make_unique<RecursiveLeastSquares>();
    adaptation_algorithm_->initialize(param_dimension_);
    
    // Set up data collection
    data_buffer_.reserve(buffer_size_);
    
    LOG_INFO("Started sim-to-real adaptation");
  }
  
  RobotData collectRealData() {
    RobotData data;
    
    // Collect state measurements
    data.state = getCurrentState();
    
    // Collect action/command data
    data.action = getLastAction();
    
    // Collect resulting state change
    data.next_state = getCurrentStateAfterDelay();
    
    // Collect time step
    data.dt = control_dt_;
    
    return data;
  }
  
  SystemState predictWithSimulation(const RobotAction& action) {
    // Apply current model parameters to simulation
    simulation_model_.setParameters(current_params_);
    
    // Predict next state given current state and action
    return simulation_model_.predict(getCurrentState(), action, control_dt_);
  }
  
  void updateModelParameters(const RobotData& real_data, 
                           const SystemState& sim_prediction) {
    // Calculate prediction error
    auto error = calculatePredictionError(real_data.next_state, sim_prediction);
    
    // Update parameter estimates using system identification
    auto param_correction = adaptation_algorithm_->update(
        real_data, error, current_params_);
    
    // Apply parameter correction
    for (int i = 0; i < param_dimension_; i++) {
      current_params_[i] += param_correction[i];
      
      // Apply bounds checking
      current_params_[i] = clamp(current_params_[i], 
                                param_bounds_[i].first, 
                                param_bounds_[i].second);
    }
    
    // Update simulation model with new parameters
    simulation_model_.setParameters(current_params_);
  }
  
  void publishAdaptedParameters() {
    auto proto = tx_adapted_params().initProto();
    
    for (int i = 0; i < param_dimension_; i++) {
      proto.add_parameters(current_params_[i]);
    }
    
    tx_adapted_params().publish();
  }

  std::unique_ptr<ParameterEstimator> param_estimator_;
  std::unique_ptr<AdaptationAlgorithm> adaptation_algorithm_;
  RobotModel simulation_model_;
  std::vector<float> simulation_params_;
  std::vector<float> current_params_;
  bool is_real_robot_ = false;
  std::string sim_model_path_ = "models/robot.urdf";
  
  // Adaptation parameters
  int param_dimension_ = 20;
  int buffer_size_ = 1000;
  float control_dt_ = 0.01f;
  std::vector<std::pair<float, float>> param_bounds_;
  
  // Communication channels
  ISAAC_PROTO_RX(StateProto, state);
  ISAAC_PROTO_RX(ActionProto, action);
  ISAAC_PROTO_TX(ParametersProto, adapted_params);
};

}  // namespace sysid
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::sysid::SimRealAdapter);
```

## Curriculum Learning

### Progressive Task Complexity
Curriculum learning involves training on progressively more difficult tasks:

```cpp
#include "engine/alice/alice.hpp"
#include "engine/gems/curriculum_learning/task_scheduler.hpp"

namespace isaac {
namespace curriculum {

class CurriculumLearning : public Codelet {
 public:
  void start() override {
    // Initialize task progression
    initializeTaskProgression();
    
    // Start with easiest task
    current_task_level_ = 0;
    switchToTask(current_task_level_);
    
    // Set up progress evaluation
    performance_evaluator_ = std::make_unique<PerformanceEvaluator>();
    
    tickPeriodically(std::chrono::seconds(30));  // Evaluate progress every 30 seconds
  }

  void tick() override {
    // Evaluate current task performance
    float performance = evaluateCurrentTaskPerformance();
    
    // Check if ready to advance to next task
    if (isReadyForNextTask(performance)) {
      advanceToNextTask();
    }
    
    // Occasionally evaluate overall curriculum progress
    if (step_count_ % evaluation_frequency_ == 0) {
      evaluateCurriculumProgress();
    }
    
    step_count_++;
  }

 private:
  void initializeTaskProgression() {
    // Define task progression
    tasks_ = {
      {"Reach to fixed target", 0.8f},  // Require 80% success rate
      {"Reach to random target", 0.85f}, 
      {"Pick up simple object", 0.75f},
      {"Place object in bin", 0.70f},
      {"Pick and place sequence", 0.65f},
      {"Multi-object manipulation", 0.60f}
    };
    
    // Define task parameters for each level
    task_params_[0]["reach_distance"] = 0.3f;  // 30cm reach
    task_params_[1]["reach_distance"] = 0.5f;  // 50cm reach with randomization
    task_params_[2]["object_size"] = 0.05f;    // 5cm cube
    task_params_[3]["target_accuracy"] = 0.02f; // 2cm placement accuracy
  }
  
  bool isReadyForNextTask(float current_performance) {
    // Check if current performance exceeds threshold
    if (current_performance >= tasks_[current_task_level_].second) {
      // Additional checks: minimum training time, consistency, etc.
      if (episodes_since_evaluation_ >= min_episodes_per_task_ &&
          performance_streak_ >= min_success_streak_) {
        return true;
      }
    }
    
    return false;
  }
  
  void advanceToNextTask() {
    if (current_task_level_ < tasks_.size() - 1) {
      current_task_level_++;
      
      // Switch to next task
      switchToTask(current_task_level_);
      
      // Reset evaluation counters
      episodes_since_evaluation_ = 0;
      performance_streak_ = 0;
      
      LOG_INFO("Advanced to task level %d: %s", 
               current_task_level_, 
               tasks_[current_task_level_].first.c_str());
    } else {
      LOG_INFO("Completed entire curriculum!");
      // Could trigger advanced training or real-world deployment
    }
  }
  
  void switchToTask(int level) {
    // Update task parameters
    updateTaskParameters(level);
    
    // Modify reward function if needed
    modifyRewardFunction(level);
    
    // Adjust environment difficulty
    adjustEnvironmentDifficulty(level);
  }
  
  void updateTaskParameters(int level) {
    // Apply task-specific parameters
    if (task_params_.count(level) > 0) {
      for (const auto& param : task_params_[level]) {
        setParameterValue(param.first, param.second);
      }
    }
  }
  
  float evaluateCurrentTaskPerformance() {
    // Calculate performance based on recent episodes
    auto recent_episodes = getRecentEpisodes(num_recent_episodes_);
    
    float success_count = 0;
    for (const auto& episode : recent_episodes) {
      if (episode.isSuccessful()) {
        success_count++;
      }
    }
    
    return success_count / recent_episodes.size();
  }
  
  void evaluateCurriculumProgress() {
    // Overall curriculum evaluation
    float overall_performance = calculateOverallPerformance();
    
    LOG_INFO("Curriculum Progress - Level: %d/%zu, Performance: %.2f%%", 
             current_task_level_ + 1, tasks_.size(), 
             overall_performance * 100.0f);
  }

  std::vector<std::pair<std::string, float>> tasks_;  // {task_name, threshold}
  std::map<int, std::map<std::string, float>> task_params_;
  int current_task_level_;
  std::unique_ptr<PerformanceEvaluator> performance_evaluator_;
  
  // Progress tracking
  int step_count_ = 0;
  int episodes_since_evaluation_ = 0;
  int performance_streak_ = 0;
  int evaluation_frequency_ = 300;  // Evaluate every 300 steps
  int min_episodes_per_task_ = 50;
  int min_success_streak_ = 5;
  int num_recent_episodes_ = 20;
};

}  // namespace curriculum
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::curriculum::CurriculumLearning);
```

## Isaac Sim for RL Training

### Isaac Sim RL Environment
Using Isaac Sim for reinforcement learning:

```python
# Python example using Isaac Sim for RL training
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.robots import Robot
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.sensor import Camera
from omni.isaac.core.tasks import BaseTask
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class RLRobotTask(BaseTask):
    def __init__(self, name):
        super().__init__(name=name, offset=np.array([0, 0, 0]))
        
        # Task parameters
        self._num_envs = 100
        self._env_spacing = 2.0
        self._max_episode_length = 500
        
        # Robot parameters
        self._robot_positions = np.zeros((self._num_envs, 3))
        self._robot_orientations = np.zeros((self._num_envs, 4))
        
        # Task-specific parameters
        self._target_positions = np.zeros((self._num_envs, 3))
        
    def set_up_scene(self, scene):
        # Set up the environment scene
        super().set_up_scene(scene)
        
        # Add robot to each environment
        for i in range(self._num_envs):
            robot_path = f"/World/envs/env_{i}/Robot"
            self._robot_positions[i] = [float(i % 10) * self._env_spacing, 
                                       float(i // 10) * self._env_spacing, 
                                       0.1]
            
            add_reference_to_stage(
                usd_path=get_assets_root_path() + "/Isaac/Robots/Franka/franka_instanceable.usd",
                prim_path=robot_path
            )
            
            # Add target object
            target_path = f"/World/envs/env_{i}/Target"
            self._target_positions[i] = [
                self._robot_positions[i][0] + np.random.uniform(-1.0, 1.0),
                self._robot_positions[i][1] + np.random.uniform(-1.0, 1.0),
                0.1
            ]
            
            scene.add(
                DynamicCuboid(
                    prim_path=target_path,
                    name=f"target_{i}",
                    position=self._target_positions[i],
                    size=0.05,
                    color=np.array([0.8, 0.1, 0.1])
                )
            )
        
        # Add robots to scene
        for i in range(self._num_envs):
            robot_path = f"/World/envs/env_{i}/Robot"
            scene.add(
                Robot(
                    prim_path=robot_path,
                    name=f"franka_{i}",
                    position=self._robot_positions[i]
                )
            )
    
    def get_observations(self):
        # Get observations for all environments
        obs_dict = {}
        
        # Robot joint positions and velocities
        joint_pos = self._robots.get_joint_positions()
        joint_vel = self._robots.get_joint_velocities()
        
        # Target positions relative to robot
        robot_pos = self._robots.get_world_poses()[0]
        rel_target_pos = self._target_positions - robot_pos
        
        # Concatenate observations
        obs = np.hstack([joint_pos, joint_vel, rel_target_pos])
        
        obs_dict["policy"] = obs
        return obs_dict
    
    def get_extras(self):
        # Additional information for logging, etc.
        extras = {}
        extras["episode_lengths"] = self._episode_lengths
        extras["rewards"] = self._rewards
        return extras
    
    def pre_physics_step(self, actions):
        # Process actions before physics step
        self._actions = actions.detach().clone()
        
        # Convert actions to joint commands
        joint_commands = self._actions.cpu().numpy()
        
        # Apply actions to robots
        self._robots.set_joint_position_targets(joint_commands)
    
    def post_reset(self):
        # Reset the task
        self._episode_lengths = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._episode_count = 0
        self._steps_count = 0

# DDPG Actor Network for continuous control
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        
    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

# DDPG Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q = torch.relu(self.l1(sa))
        q = torch.relu(self.l2(q))
        q = self.l3(q)
        return q

# DDPG Agent Implementation
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.max_action = max_action
        self.discount = 0.99
        self.tau = 0.005
        
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self, replay_buffer, batch_size=100):
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        reward = torch.FloatTensor(reward).to(device)
        not_done = torch.FloatTensor(not_done).to(device)
        
        # Compute target Q-value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()
        
        # Get current Q-value estimate
        current_Q = self.critic(state, action)
        
        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# Training loop
def train_rl_agent():
    # Initialize Isaac Sim world
    my_world = World(stage_units_in_meters=1.0, rendering_dt=1.0/60.0, sim_params={"use_gpu": True})
    
    # Create task
    my_task = RLRobotTask(name="rl_task")
    my_world.add_task(my_task)
    
    # Reset world
    my_world.reset()
    
    # Get robot from world
    robots = my_world.scene.get_object("franka_0")  # Get first robot
    
    # Initialize RL agent
    state_dim = 20  # Example state dimension
    action_dim = 7  # Example action dimension (Franka arm joints)
    max_action = 1.0
    agent = DDPGAgent(state_dim, action_dim, max_action)
    
    # Training parameters
    max_timesteps = 1000000
    start_timesteps = 10000
    eval_freq = 5000
    
    state, done = my_world.get_observations()["policy"][0], False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    
    for t in range(max_timesteps):
        episode_timesteps += 1
        
        # Select action randomly or from policy
        if t < start_timesteps:
            action = np.random.normal(0, max_action, size=action_dim)
        else:
            action = agent.select_action(np.array(state))
            # Add exploration noise
            noise = np.random.normal(0, max_action * 0.1, size=action_dim)
            action = (action + noise).clip(-max_action, max_action)
        
        # Perform action in Isaac Sim
        my_world.step(render=True)
        
        # Get new state and reward
        reward, done = get_reward_from_sim()  # Custom reward function
        next_state = my_world.get_observations()["policy"][0]
        
        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done)
        
        state = next_state
        episode_reward += reward
        
        # Train agent after collecting sufficient data
        if t >= start_timesteps:
            agent.train(replay_buffer)
        
        # Check if episode is done
        if done:
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            my_world.reset()
            state, done = my_world.get_observations()["policy"][0], False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
```

## Evaluation and Validation

### Assessing Sim-to-Real Transfer
Methods for evaluating the effectiveness of sim-to-real transfer:

```cpp
#include "engine/alice/alice.hpp"
#include "engine/gems/evaluation/performance_metrics.hpp"

namespace isaac {
namespace evaluation {

class SimRealEvaluator : public Codelet {
 public:
  void start() override {
    // Initialize evaluation metrics
    initializeMetrics();
    
    // Set up data collection
    setupDataCollection();
    
    // Start evaluation
    evaluation_active_ = true;
    
    tickPeriodically(std::chrono::seconds(1));  // Evaluate every second
  }

  void tick() override {
    if (evaluation_active_) {
      // Collect performance data
      collectPerformanceData();
      
      // Calculate metrics
      calculateTransferMetrics();
      
      // Log results
      logEvaluationResults();
    }
  }

 private:
  void initializeMetrics() {
    // Define metrics to track
    metrics_["tracking_error"] = 0.0f;
    metrics_["execution_time"] = 0.0f;
    metrics_["success_rate"] = 0.0f;
    metrics_["energy_efficiency"] = 0.0f;
    metrics_["robustness"] = 0.0f;
  }
  
  void collectPerformanceData() {
    // Collect data from both sim and real
    if (has_sim_data_) {
      auto sim_data = rx_sim_data().popLatest();
      if (sim_data) {
        sim_performance_data_.push_back(sim_data.value());
      }
    }
    
    if (has_real_data_) {
      auto real_data = rx_real_data().popLatest();
      if (real_data) {
        real_performance_data_.push_back(real_data.value());
      }
    }
  }
  
  void calculateTransferMetrics() {
    // Calculate sim-to-real gap
    float tracking_error = calculateTrackingError();
    float execution_time_ratio = calculateExecutionTimeRatio();
    float success_rate_drop = calculateSuccessRateDrop();
    
    // Update metrics
    metrics_["tracking_error"] = tracking_error;
    metrics_["execution_time_ratio"] = execution_time_ratio;
    metrics_["success_rate_drop"] = success_rate_drop;
    
    // Calculate overall transfer score
    float transfer_score = calculateTransferScore();
    metrics_["transfer_score"] = transfer_score;
  }
  
  float calculateTrackingError() {
    // Calculate average deviation between sim and real trajectories
    float total_error = 0.0f;
    int count = 0;
    
    for (size_t i = 0; i < std::min(sim_performance_data_.size(), 
                                    real_performance_data_.size()); i++) {
      auto sim_traj = sim_performance_data_[i].trajectory;
      auto real_traj = real_performance_data_[i].trajectory;
      
      // Calculate trajectory deviation
      float traj_error = calculateTrajectoryDeviation(sim_traj, real_traj);
      total_error += traj_error;
      count++;
    }
    
    return count > 0 ? total_error / count : 0.0f;
  }
  
  float calculateTransferScore() {
    // Weighted combination of different metrics
    float score = 0.0f;
    
    // Lower tracking error is better
    score += (1.0f / (1.0f + metrics_["tracking_error"])) * 0.3f;
    
    // Execution time close to 1.0 is better
    float time_efficiency = 1.0f / (1.0f + abs(metrics_["execution_time_ratio"] - 1.0f));
    score += time_efficiency * 0.2f;
    
    // Higher success rate is better
    score += metrics_["success_rate"] * 0.3f;
    
    // Higher energy efficiency is better
    score += metrics_["energy_efficiency"] * 0.2f;
    
    return score;
  }
  
  void logEvaluationResults() {
    LOG_INFO("Sim-to-Real Evaluation Results:");
    LOG_INFO("  Tracking Error: %.3f", metrics_["tracking_error"]);
    LOG_INFO("  Execution Time Ratio: %.3f", metrics_["execution_time_ratio"]);
    LOG_INFO("  Success Rate Drop: %.3f", metrics_["success_rate_drop"]);
    LOG_INFO("  Transfer Score: %.3f", metrics_["transfer_score"]);
  }

  std::map<std::string, float> metrics_;
  std::vector<PerformanceData> sim_performance_data_;
  std::vector<PerformanceData> real_performance_data_;
  bool evaluation_active_ = false;
  bool has_sim_data_ = true;
  bool has_real_data_ = true;
  
  // Communication channels
  ISAAC_PROTO_RX(PerformanceDataProto, sim_data);
  ISAAC_PROTO_RX(PerformanceDataProto, real_data);
  ISAAC_PROTO_TX(EvaluationResultsProto, evaluation_results);
};

}  // namespace evaluation
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::evaluation::SimRealEvaluator);
```

## Best Practices

### Effective RL for Robotics
1. **Reward Shaping**: Design rewards that guide learning toward desired behaviors
2. **Exploration Strategies**: Use appropriate exploration methods for continuous control
3. **Sample Efficiency**: Implement techniques to maximize learning from limited samples
4. **Safety Constraints**: Incorporate safety considerations into the learning process

### Sim-to-Real Transfer
1. **Domain Randomization**: Extensively randomize simulation parameters
2. **System Identification**: Adapt models based on real-world data
3. **Robust Control**: Design policies that are robust to model inaccuracies
4. **Gradual Deployment**: Progressively increase task difficulty in reality

### Validation and Testing
1. **Comprehensive Evaluation**: Test across multiple scenarios and conditions
2. **Safety Testing**: Ensure safe behavior under unexpected conditions
3. **Long-term Stability**: Verify performance over extended operation periods
4. **Comparison Baselines**: Compare against traditional control methods

## Summary

Reinforcement learning provides a powerful approach for learning complex robotic behaviors, especially when combined with simulation for safe and efficient training. The key to successful sim-to-real transfer lies in careful domain randomization, system identification, and progressive validation. By following best practices in RL for robotics, we can develop adaptive and capable robotic systems.

## Exercises

1. Implement a simple RL algorithm for a robotic control task in simulation.
2. Apply domain randomization to improve sim-to-real transfer.
3. Design a curriculum learning approach for a complex manipulation task.
4. Evaluate the effectiveness of sim-to-real transfer using appropriate metrics.