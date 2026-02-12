---
sidebar_position: 2
---

# Week 12: Bipedal Locomotion and Balance Control

## Learning Objectives

By the end of this week, you will be able to:
- Understand the principles of bipedal locomotion
- Implement dynamic walking controllers for humanoid robots
- Design balance control strategies for perturbation recovery
- Analyze the stability of walking gaits
- Implement stepping strategies for balance recovery
- Understand the role of sensory feedback in locomotion
- Evaluate walking performance metrics

## Principles of Bipedal Locomotion

### Differences from Quadrupedal Locomotion
Bipedal locomotion presents unique challenges compared to multi-legged locomotion:
- **Smaller support polygon**: Only two feet instead of four
- **Dynamic balance**: Constantly moving between stable states
- **Higher center of mass**: Greater instability during movement
- **Complex coordination**: Requires precise timing between upper and lower body

### Phases of Walking
Human walking consists of two main phases:
- **Single Support Phase (SSP)**: One foot is on the ground
- **Double Support Phase (DSP)**: Both feet are on the ground

During SSP, the stance leg supports the body weight while the swing leg moves forward. During DSP, weight transfers from the trailing leg to the leading leg.

### Walking Pattern Parameters
```cpp
#include "engine/alice/alice.hpp"
#include "engine/gems/walking/locomotion_controller.hpp"

namespace isaac {
namespace humanoid {

class WalkingController : public Codelet {
 public:
  void start() override {
    // Initialize walking parameters
    initializeWalkingParameters();
    
    // Set up gait phase detection
    setupGaitPhaseDetection();
    
    // Initialize balance control systems
    initializeBalanceSystems();
    
    tickPeriodically(std::chrono::milliseconds(10));  // 100 Hz control
  }

  void tick() override {
    // Get current state
    auto joint_states = rx_joint_states().popLatest().value();
    auto imu_data = rx_imu_data().popLatest().value();
    auto ft_sensors = rx_force_torque_sensors().popLatest().value();
    
    // Update gait phase
    updateGaitPhase();
    
    // Compute desired walking motion
    auto desired_motion = computeDesiredWalkingMotion();
    
    // Apply balance control
    auto balance_correction = computeBalanceCorrection(imu_data, ft_sensors);
    
    // Generate joint commands
    auto joint_commands = generateJointCommands(desired_motion, balance_correction);
    
    // Publish commands
    publishJointCommands(joint_commands);
    
    // Monitor walking performance
    monitorWalkingPerformance();
  }

 private:
  void initializeWalkingParameters() {
    // Basic walking parameters
    step_length_ = 0.30;        // 30 cm step length
    step_width_ = 0.20;         // 20 cm step width
    step_height_ = 0.05;        // 5 cm step height
    step_duration_ = 1.0;       // 1 second per step
    dsp_ratio_ = 0.1;          // 10% double support
    com_height_ = 0.85;        // 85 cm COM height
    
    // Walking speed parameters
    max_forward_speed_ = 0.5;   // 0.5 m/s max
    max_turn_speed_ = 0.3;      // 0.3 rad/s max
    max_lateral_speed_ = 0.2;   // 0.2 m/s max
    
    // Timing parameters
    control_frequency_ = 100.0; // 100 Hz
    control_dt_ = 1.0 / control_frequency_;
  }
  
  void setupGaitPhaseDetection() {
    // Initialize phase tracking
    current_phase_ = GaitPhase::DOUBLE_SUPPORT;
    phase_start_time_ = getNodeTime();
    step_count_ = 0;
    
    // Set up phase transition detection
    ssp_duration_ = step_duration_ * (1.0 - dsp_ratio_);
    dsp_duration_ = step_duration_ * dsp_ratio_;
  }
  
  void updateGaitPhase() {
    double elapsed_time = getNodeTime() - phase_start_time_;
    
    switch (current_phase_) {
      case GaitPhase::DOUBLE_SUPPORT:
        if (elapsed_time > dsp_duration_) {
          current_phase_ = GaitPhase::SINGLE_SUPPORT_LEFT;
          phase_start_time_ = getNodeTime();
          step_count_++;
        }
        break;
        
      case GaitPhase::SINGLE_SUPPORT_LEFT:
        if (elapsed_time > ssp_duration_) {
          current_phase_ = GaitPhase::DOUBLE_SUPPORT;
          phase_start_time_ = getNodeTime();
        }
        break;
        
      case GaitPhase::SINGLE_SUPPORT_RIGHT:
        if (elapsed_time > ssp_duration_) {
          current_phase_ = GaitPhase::DOUBLE_SUPPORT;
          phase_start_time_ = getNodeTime();
        }
        break;
    }
  }
  
  WalkingMotion computeDesiredWalkingMotion() {
    WalkingMotion motion;
    
    // Calculate desired step based on command
    if (walking_command_.has_value()) {
      auto cmd = walking_command_.value();
      
      // Calculate step parameters based on desired velocity
      double forward_step = cmd.forward_velocity * step_duration_;
      double lateral_step = cmd.lateral_velocity * step_duration_;
      double turn_step = cmd.turn_velocity * step_duration_;
      
      // Apply limits
      forward_step = clamp(forward_step, -step_length_, step_length_);
      lateral_step = clamp(lateral_step, -step_width_/2, step_width_/2);
      turn_step = clamp(turn_step, -0.2, 0.2);  // Limit turning
      
      motion.forward_step = forward_step;
      motion.lateral_step = lateral_step;
      motion.turn_step = turn_step;
    } else {
      // Default to standing posture
      motion.forward_step = 0.0;
      motion.lateral_step = 0.0;
      motion.turn_step = 0.0;
    }
    
    return motion;
  }

  enum class GaitPhase {
    DOUBLE_SUPPORT,
    SINGLE_SUPPORT_LEFT,
    SINGLE_SUPPORT_RIGHT
  };
  
  // Walking parameters
  double step_length_;
  double step_width_;
  double step_height_;
  double step_duration_;
  double dsp_ratio_;
  double com_height_;
  double max_forward_speed_;
  double max_turn_speed_;
  double max_lateral_speed_;
  double control_frequency_;
  double control_dt_;
  double ssp_duration_;
  double dsp_duration_;
  
  // Gait phase tracking
  GaitPhase current_phase_;
  double phase_start_time_;
  int step_count_;
  
  // Commands
  std::optional<WalkingCommand> walking_command_;
  
  // Communication channels
  ISAAC_PROTO_RX(JointStatesProto, joint_states);
  ISAAC_PROTO_RX(ImuDataProto, imu_data);
  ISAAC_PROTO_RX(ForceTorqueProto, force_torque_sensors);
  ISAAC_PROTO_RX(WalkingCommandProto, walking_command);
  ISAAC_PROTO_TX(JointCommandsProto, joint_commands);
};

}  // namespace humanoid
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::humanoid::WalkingController);
```

## Dynamic Walking Controllers

### Spring-Loaded Inverted Pendulum (SLIP) Model
The SLIP model is a simplified representation of legged locomotion:

```cpp
#include "engine/alice/alice.hpp"
#include "engine/gems/locomotion/slip_model.hpp"

namespace isaac {
namespace locomotion {

class SlipWalkingController : public Codelet {
 public:
  void start() override {
    // Initialize SLIP model parameters
    initializeSlipParameters();
    
    // Set up stance and flight phase controllers
    setupPhaseControllers();
    
    tickPeriodically(std::chrono::milliseconds(1));  // High frequency for dynamics
  }

  void tick() override {
    // Get current state
    auto robot_state = rx_robot_state().popLatest().value();
    
    // Determine contact state
    bool left_contact = isLeftFootInContact();
    bool right_contact = isRightFootInContact();
    
    // Update SLIP model based on contact state
    if (left_contact && !right_contact) {
      // Left leg stance
      updateStancePhase(SupportLeg::LEFT, robot_state);
    } else if (right_contact && !left_contact) {
      // Right leg stance
      updateStancePhase(SupportLeg::RIGHT, robot_state);
    } else if (left_contact && right_contact) {
      // Double support
      updateDoubleSupportPhase(robot_state);
    } else {
      // Flight phase (shouldn't happen in normal walking)
      updateFlightPhase(robot_state);
    }
    
    // Generate control commands
    auto commands = generateControlCommands();
    publishControlCommands(commands);
  }

 private:
  void initializeSlipParameters() {
    // SLIP model parameters
    body_mass_ = 75.0;           // kg
    leg_rest_length_ = 0.9;      // m
    leg_stiffness_ = 20000.0;    // N/m
    gravity_ = 9.81;            // m/s^2
    com_height_ = 0.85;         // m (initial guess)
    
    // Controller parameters
    k_p_ = 1000.0;              // Position gain
    k_d_ = 200.0;               // Damping gain
    k_theta_ = 500.0;           // Angle gain
    
    // Initialize state
    current_apex_ = {0.0, 0.0, com_height_};
    current_velocity_ = {0.0, 0.0, 0.0};
    current_leg_angle_ = 0.0;
  }
  
  void updateStancePhase(SupportLeg support_leg, const RobotState& state) {
    // Get support leg position and orientation
    Vector3d support_foot_pos = (support_leg == SupportLeg::LEFT) ? 
                                state.left_foot_position : 
                                state.right_foot_position;
    
    // Calculate leg vector (from COM to foot)
    Vector3d leg_vector = support_foot_pos - state.com_position;
    double leg_length = leg_vector.norm();
    Vector3d leg_direction = leg_vector.normalized();
    
    // Calculate spring force
    double compression = leg_rest_length_ - leg_length;
    Vector3d spring_force = leg_stiffness_ * compression * leg_direction;
    
    // Add damping force
    Vector3d foot_velocity = (support_leg == SupportLeg::LEFT) ? 
                             state.left_foot_velocity : 
                             state.right_foot_velocity;
    Vector3d body_velocity = state.com_velocity;
    Vector3d relative_velocity = body_velocity - foot_velocity;
    Vector3d damping_force = -k_d_ * relative_velocity;
    
    // Total force on body
    Vector3d total_force = spring_force + damping_force;
    total_force.z() -= body_mass_ * gravity_;  // Add gravity
    
    // Calculate acceleration
    Vector3d acceleration = total_force / body_mass_;
    
    // Update state (numerical integration)
    current_velocity_ += acceleration * control_dt_;
    current_apex_ += current_velocity_ * control_dt_;
    
    // Calculate desired joint torques to achieve this motion
    computeJointTorques(support_leg, state, acceleration);
  }
  
  void computeJointTorques(SupportLeg support_leg, 
                          const RobotState& state, 
                          const Vector3d& desired_accel) {
    // Use inverse dynamics to compute required joint torques
    // This is a simplified version - in practice, use full inverse dynamics
    
    // Calculate desired joint accelerations to achieve body acceleration
    auto joint_acc = computeRequiredJointAcc(state, desired_accel);
    
    // Apply PD control to track desired accelerations
    auto current_joints = state.joint_positions;
    auto current_velocities = state.joint_velocities;
    
    auto joint_torques = joint_stiffness_ * (joint_acc - current_joints) - 
                         joint_damping_ * current_velocities;
    
    // Publish torques
    publishJointTorques(joint_torques);
  }
  
  void updateDoubleSupportPhase(const RobotState& state) {
    // In double support, average the forces from both legs
    Vector3d left_foot_pos = state.left_foot_position;
    Vector3d right_foot_pos = state.right_foot_position;
    Vector3d com_pos = state.com_position;
    
    // Calculate vectors to each foot
    Vector3d left_leg_vec = left_foot_pos - com_pos;
    Vector3d right_leg_vec = right_foot_pos - com_pos;
    
    // Calculate forces from each leg
    double left_compression = leg_rest_length_ - left_leg_vec.norm();
    double right_compression = leg_rest_length_ - right_leg_vec.norm();
    
    Vector3d left_force = leg_stiffness_ * left_compression * left_leg_vec.normalized();
    Vector3d right_force = leg_stiffness_ * right_compression * right_leg_vec.normalized();
    
    // Total force
    Vector3d total_force = left_force + right_force;
    total_force.z() -= body_mass_ * gravity_;
    
    // Update state
    Vector3d acceleration = total_force / body_mass_;
    current_velocity_ += acceleration * control_dt_;
    current_apex_ += current_velocity_ * control_dt_;
  }

  enum class SupportLeg { LEFT, RIGHT };
  
  // SLIP parameters
  double body_mass_;
  double leg_rest_length_;
  double leg_stiffness_;
  double gravity_;
  double com_height_;
  
  // Controller gains
  double k_p_;
  double k_d_;
  double k_theta_;
  
  // State
  Vector3d current_apex_;
  Vector3d current_velocity_;
  double current_leg_angle_;
  
  // Time parameters
  double control_dt_ = 0.001;  // 1kHz control
  
  // Joint control parameters
  VectorXd joint_stiffness_;
  VectorXd joint_damping_;
  
  // Communication channels
  ISAAC_PROTO_RX(RobotStateProto, robot_state);
  ISAAC_PROTO_TX(JointTorquesProto, joint_torques);
};

}  // namespace locomotion
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::locomotion::SlipWalkingController);
```

## Balance Control Strategies

### Ankle-Hip-Arm Coordination
Effective balance control requires coordination across multiple strategies:

```cpp
#include "engine/alice/alice.hpp"
#include "engine/gems/control/balance_coordinator.hpp"

namespace isaac {
namespace balance {

class MultiStrategyBalancer : public Codelet {
 public:
  void start() override {
    // Initialize individual balance strategies
    initializeAnkleStrategy();
    initializeHipStrategy();
    initializeArmStrategy();
    initializeSteppingStrategy();
    
    // Set up strategy coordinator
    setupStrategyCoordinator();
    
    tickPeriodically(std::chrono::milliseconds(10));  // 100 Hz
  }

  void tick() override {
    // Get current state
    auto imu_data = rx_imu_data().popLatest().value();
    auto joint_states = rx_joint_states().popLatest().value();
    auto ft_sensors = rx_force_torque_sensors().popLatest().value();
    
    // Calculate disturbance magnitude
    double disturbance_mag = calculateDisturbanceMagnitude(imu_data);
    
    // Determine required balance strategy based on disturbance
    BalanceStrategy required_strategy = determineRequiredStrategy(disturbance_mag);
    
    // Execute appropriate strategy
    switch (required_strategy) {
      case BalanceStrategy::ANKLE:
        executeAnkleStrategy(imu_data, joint_states);
        break;
      case BalanceStrategy::HIP:
        executeHipStrategy(imu_data, joint_states);
        break;
      case BalanceStrategy::ARM:
        executeArmStrategy(imu_data, joint_states);
        break;
      case BalanceStrategy::STEPPING:
        executeSteppingStrategy(imu_data, joint_states, ft_sensors);
        break;
      case BalanceStrategy::COOPERATIVE:
        executeCooperativeStrategy(imu_data, joint_states, ft_sensors);
        break;
    }
    
    // Monitor balance performance
    monitorBalancePerformance();
  }

 private:
  void initializeAnkleStrategy() {
    ankle_params_.kp = 200.0;      // Position gain
    ankle_params_.kd = 10.0;       // Damping gain
    ankle_params_.max_angle = 0.1; // 5.7 degrees max
    ankle_params_.cutoff_freq = 10.0; // 10 Hz low-pass filter
  }
  
  void initializeHipStrategy() {
    hip_params_.kp = 500.0;
    hip_params_.kd = 50.0;
    hip_params_.max_angle = 0.2;   // 11.4 degrees max
    hip_params_.torso_weight = 0.7; // How much torso to move
  }
  
  void initializeArmStrategy() {
    arm_params_.kp = 100.0;
    arm_params_.kd = 5.0;
    arm_params_.swing_gain = 50.0; // Gain for arm swinging
    arm_params_.max_extension = 0.3; // Max arm extension
  }
  
  void initializeSteppingStrategy() {
    step_params_.reaction_time = 0.2;  // 200ms reaction time
    step_params_.max_step_size = 0.4;  // 40cm max step
    step_params_.step_height = 0.1;    // 10cm step height
    step_params_.step_duration = 0.8;  // 800ms step time
  }
  
  BalanceStrategy determineRequiredStrategy(double disturbance) {
    if (disturbance < ankle_threshold_) {
      return BalanceStrategy::ANKLE;
    } else if (disturbance < hip_threshold_) {
      return BalanceStrategy::HIP;
    } else if (disturbance < arm_threshold_) {
      return BalanceStrategy::ARM;
    } else if (disturbance < step_threshold_) {
      return BalanceStrategy::STEPPING;
    } else {
      return BalanceStrategy::COOPERATIVE;
    }
  }
  
  void executeAnkleStrategy(const ImuData& imu, const JointStates& joints) {
    // Simple ankle strategy: lean opposite to angular velocity
    double desired_ankle_roll = -ankle_params_.kp * imu.angular_velocity.y() * control_dt_;
    double desired_ankle_pitch = -ankle_params_.kp * imu.angular_velocity.x() * control_dt_;
    
    // Apply limits
    desired_ankle_roll = clamp(desired_ankle_roll, -ankle_params_.max_angle, ankle_params_.max_angle);
    desired_ankle_pitch = clamp(desired_ankle_pitch, -ankle_params_.max_angle, ankle_params_.max_angle);
    
    // Generate ankle commands
    AnkleCommands ankle_cmd;
    ankle_cmd.left_roll = desired_ankle_roll;
    ankle_cmd.left_pitch = desired_ankle_pitch;
    ankle_cmd.right_roll = desired_ankle_roll;
    ankle_cmd.right_pitch = desired_ankle_pitch;
    
    publishAnkleCommands(ankle_cmd);
  }
  
  void executeHipStrategy(const ImuData& imu, const JointStates& joints) {
    // Hip strategy: move hips to counteract COM displacement
    double com_displacement = estimateCOMDisplacement();
    
    // Move hip opposite to COM displacement
    double desired_hip_roll = -hip_params_.torso_weight * com_displacement.y();
    double desired_hip_pitch = -hip_params_.torso_weight * com_displacement.x();
    
    // Apply hip joint commands
    JointCommands hip_cmds = generateHipCommands(desired_hip_roll, desired_hip_pitch);
    publishJointCommands(hip_cmds);
  }
  
  void executeArmStrategy(const ImuData& imu, const JointStates& joints) {
    // Arm strategy: swing arms to generate corrective angular momentum
    double com_velocity_x = estimateCOMVelocity().x();
    double com_velocity_y = estimateCOMVelocity().y();
    
    // Generate arm movements opposite to COM velocity
    double left_arm_cmd = -arm_params_.swing_gain * com_velocity_y;
    double right_arm_cmd = arm_params_.swing_gain * com_velocity_y;
    
    // Apply arm commands
    JointCommands arm_cmds = generateArmCommands(left_arm_cmd, right_arm_cmd);
    publishJointCommands(arm_cmds);
  }
  
  void executeSteppingStrategy(const ImuData& imu, 
                              const JointStates& joints, 
                              const ForceTorqueData& ft) {
    // Determine if stepping is needed
    Vector2d zmp = calculateZMP();
    Vector2d com = estimateCOMPosition();
    
    // Check if ZMP is outside support polygon
    if (!isZMPInSupportPolygon(zmp)) {
      // Plan a step to expand support polygon
      Vector2d step_location = planStepLocation(zmp, com);
      
      // Execute step
      executeStep(step_location);
    }
  }
  
  void executeCooperativeStrategy(const ImuData& imu,
                                const JointStates& joints,
                                const ForceTorqueData& ft) {
    // Combine multiple strategies cooperatively
    auto ankle_cmd = calculateAnkleCommand(imu);
    auto hip_cmd = calculateHipCommand(imu, joints);
    auto arm_cmd = calculateArmCommand(imu, joints);
    
    // Blend commands based on effectiveness
    JointCommands blended_cmd = blendCommands(ankle_cmd, hip_cmd, arm_cmd);
    publishJointCommands(blended_cmd);
  }
  
  double calculateDisturbanceMagnitude(const ImuData& imu) {
    // Calculate disturbance based on angular velocity and acceleration
    double ang_vel_mag = sqrt(pow(imu.angular_velocity.x(), 2) + 
                             pow(imu.angular_velocity.y(), 2));
    double lin_acc_mag = sqrt(pow(imu.linear_acceleration.x(), 2) + 
                             pow(imu.linear_acceleration.y(), 2));
    
    return 0.7 * ang_vel_mag + 0.3 * lin_acc_mag;
  }

  enum class BalanceStrategy {
    ANKLE,
    HIP,
    ARM,
    STEPPING,
    COOPERATIVE
  };
  
  // Strategy parameters
  struct AnkleParams {
    double kp, kd;
    double max_angle;
    double cutoff_freq;
  } ankle_params_;
  
  struct HipParams {
    double kp, kd;
    double max_angle;
    double torso_weight;
  } hip_params_;
  
  struct ArmParams {
    double kp, kd;
    double swing_gain;
    double max_extension;
  } arm_params_;
  
  struct StepParams {
    double reaction_time;
    double max_step_size;
    double step_height;
    double step_duration;
  } step_params_;
  
  // Thresholds for strategy selection
  double ankle_threshold_ = 0.1;
  double hip_threshold_ = 0.3;
  double arm_threshold_ = 0.5;
  double step_threshold_ = 0.8;
  
  // Control parameters
  double control_dt_ = 0.01;
  
  // Communication channels
  ISAAC_PROTO_RX(ImuDataProto, imu_data);
  ISAAC_PROTO_RX(JointStatesProto, joint_states);
  ISAAC_PROTO_RX(ForceTorqueProto, force_torque_sensors);
  ISAAC_PROTO_TX(JointCommandsProto, joint_commands);
  ISAAC_PROTO_TX(AnkleCommandsProto, ankle_commands);
};

}  // namespace balance
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::balance::MultiStrategyBalancer);
```

## Stepping Strategies for Balance Recovery

### Reactive Stepping
When balance is lost, stepping is often the most effective recovery strategy:

```cpp
#include "engine/alice/alice.hpp"
#include "engine/gems/stepping/recovery_stepper.hpp"

namespace isaac {
namespace stepping {

class RecoveryStepper : public Codelet {
 public:
  void start() override {
    // Initialize stepping parameters
    initializeSteppingParameters();
    
    // Set up fall detection
    setupFallDetection();
    
    tickPeriodically(std::chrono::milliseconds(10));
  }

  void tick() override {
    // Get current state
    auto imu_data = rx_imu_data().popLatest().value();
    auto joint_states = rx_joint_states().popLatest().value();
    auto com_state = rx_com_state().popLatest().value();
    
    // Check for balance loss
    bool balance_lost = detectBalanceLoss(imu_data, com_state);
    
    if (balance_lost && !currently_stepping_) {
      // Plan emergency step
      Vector2d step_location = planEmergencyStep(imu_data, com_state);
      
      // Execute step
      executeStep(step_location, EmergencyStepType::RECOVERY);
    }
    
    // Monitor ongoing steps
    monitorStepExecution();
  }

 private:
  void initializeSteppingParameters() {
    // Step planning parameters
    max_step_length_ = 0.4;      // 40cm max step
    max_step_width_ = 0.3;       // 30cm max width adjustment
    step_height_ = 0.1;          // 10cm step height
    step_duration_ = 0.8;        // 800ms step time
    toe_off_delay_ = 0.1;        // 100ms delay before lifting foot
    heel_strike_early_ = 0.1;    // 100ms early heel strike
    
    // Balance loss detection
    angular_velocity_threshold_ = 0.8;  // rad/s
    com_deviation_threshold_ = 0.15;    // 15cm from base of support
    zmp_margin_threshold_ = 0.05;       // 5cm margin from edge
  }
  
  bool detectBalanceLoss(const ImuData& imu, const ComState& com) {
    // Multiple criteria for balance loss detection
    
    // Criterion 1: High angular velocity
    double ang_vel_mag = sqrt(pow(imu.angular_velocity.x(), 2) + 
                             pow(imu.angular_velocity.y(), 2));
    bool high_ang_vel = ang_vel_mag > angular_velocity_threshold_;
    
    // Criterion 2: COM outside safe region
    Vector2d zmp = calculateCurrentZMP();
    Vector2d com_proj = com.position.head<2>();
    bool com_outside_safe = !isCOMInSafeRegion(com_proj, zmp);
    
    // Criterion 3: ZMP approaching support boundary
    bool zmp_approaching_boundary = !isZMPInSafeMargin(zmp);
    
    // Combine criteria with hysteresis
    if (high_ang_vel || com_outside_safe || zmp_approaching_boundary) {
      if (!balance_loss_detected_) {
        balance_loss_start_time_ = getNodeTime();
        balance_loss_detected_ = true;
      }
    } else {
      balance_loss_detected_ = false;
    }
    
    // Require sustained violation for some time
    if (balance_loss_detected_) {
      return (getNodeTime() - balance_loss_start_time_) > 0.05; // 50ms hysteresis
    }
    
    return false;
  }
  
  Vector2d planEmergencyStep(const ImuData& imu, const ComState& com) {
    // Use Capture Point (Capture Point = COM position + COM velocity * sqrt(leg_length / gravity))
    double leg_length = estimateLegLength();
    double time_constant = sqrt(leg_length / gravity_);
    
    Vector2d com_vel = com.velocity.head<2>();
    Vector2d capture_point = com.position.head<2>() + com_vel * time_constant;
    
    // Plan step to reach capture point
    Vector2d current_support_center = getCurrentSupportCenter();
    Vector2d desired_step = capture_point - current_support_center;
    
    // Apply limits
    double step_mag = desired_step.norm();
    if (step_mag > max_step_length_) {
      desired_step = desired_step.normalized() * max_step_length_;
    }
    
    // Ensure step is in appropriate direction relative to COM state
    Vector2d com_to_support = current_support_center - com.position.head<2>();
    double dot_product = com_to_support.dot(desired_step);
    
    if (dot_product < 0) {
      // Step in direction away from COM to expand support
      desired_step = com_to_support.normalized() * max_step_length_;
    }
    
    return current_support_center + desired_step;
  }
  
  void executeStep(const Vector2d& target_location, EmergencyStepType step_type) {
    // Generate step trajectory
    auto step_trajectory = generateStepTrajectory(target_location, step_type);
    
    // Execute step using stepping controller
    stepping_controller_.executeStep(step_trajectory);
    
    currently_stepping_ = true;
    step_start_time_ = getNodeTime();
  }
  
  StepTrajectory generateStepTrajectory(const Vector2d& target, EmergencyStepType step_type) {
    StepTrajectory traj;
    
    // Current foot position
    Vector3d current_pos = getCurrentSwingFootPosition();
    Vector2d current_xy = {current_pos.x(), current_pos.y()};
    
    // Calculate step vector
    Vector2d step_vector = target - current_xy;
    double step_distance = step_vector.norm();
    
    // Generate swing trajectory
    int num_points = static_cast<int>(step_duration_ / control_dt_);
    traj.points.resize(num_points);
    
    for (int i = 0; i < num_points; i++) {
      double t = static_cast<double>(i) / num_points;
      
      // Horizontal interpolation
      Vector2d xy_pos = current_xy + t * step_vector;
      
      // Vertical profile (parabolic lift and lower)
      double z_pos;
      if (t < 0.3) {
        // Lift phase (first 30%)
        double lift_t = t / 0.3;
        z_pos = current_pos.z() + step_height_ * pow(sin(lift_t * M_PI / 2), 2);
      } else if (t > 0.7) {
        // Lower phase (last 30%)
        double lower_t = (t - 0.7) / 0.3;
        z_pos = current_pos.z() + step_height_ * pow(cos(lower_t * M_PI / 2), 2);
      } else {
        // Cruise phase (middle 40%)
        z_pos = current_pos.z() + step_height_;
      }
      
      traj.points[i] = {xy_pos.x(), xy_pos.y(), z_pos};
    }
    
    return traj;
  }
  
  void monitorStepExecution() {
    if (currently_stepping_) {
      // Check if step is complete
      if (getNodeTime() - step_start_time_ > step_duration_) {
        currently_stepping_ = false;
        
        // Verify step was successful
        if (isBalanceRecovered()) {
          LOG_INFO("Balance recovered with emergency step");
        } else {
          LOG_WARN("Emergency step did not recover balance");
        }
      }
    }
  }

  // Stepping parameters
  double max_step_length_;
  double max_step_width_;
  double step_height_;
  double step_duration_;
  double toe_off_delay_;
  double heel_strike_early_;
  double angular_velocity_threshold_;
  double com_deviation_threshold_;
  double zmp_margin_threshold_;
  double gravity_ = 9.81;
  double control_dt_ = 0.01;
  
  // State tracking
  bool balance_loss_detected_ = false;
  bool currently_stepping_ = false;
  double balance_loss_start_time_ = 0.0;
  double step_start_time_ = 0.0;
  
  // Stepping controller
  SteppingController stepping_controller_;
  
  // Communication channels
  ISAAC_PROTO_RX(ImuDataProto, imu_data);
  ISAAC_PROTO_RX(JointStatesProto, joint_states);
  ISAAC_PROTO_RX(ComStateProto, com_state);
  ISAAC_PROTO_TX(StepCommandProto, step_command);
};

}  // namespace stepping
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::stepping::RecoveryStepper);
```

## Sensory Feedback Integration

### Sensor Fusion for Locomotion
Integrating multiple sensors for robust locomotion:

```cpp
#include "engine/alice/alice.hpp"
#include "engine/gems/sensor_fusion/locomotion_fusion.hpp"

namespace isaac {
namespace fusion {

class LocomotionSensorFusion : public Codelet {
 public:
  void start() override {
    // Initialize sensor models
    initializeIMUModel();
    initializeForceTorqueModel();
    initializeJointEncoderModel();
    initializeVisionModel();
    
    // Set up Kalman filter for state estimation
    setupStateEstimator();
    
    tickPeriodically(std::chrono::milliseconds(1));
  }

  void tick() override {
    // Get all sensor measurements
    auto imu_meas = rx_imu_measurements().popLatest();
    auto ft_meas = rx_force_torque_measurements().popLatest();
    auto joint_meas = rx_joint_encoder_measurements().popLatest();
    auto vision_meas = rx_vision_measurements().popLatest();
    
    // Update state estimate with new measurements
    if (imu_meas) {
      updateWithIMU(imu_meas.value());
    }
    
    if (ft_meas) {
      updateWithForceTorque(ft_meas.value());
    }
    
    if (joint_meas) {
      updateWithJointEncoders(joint_meas.value());
    }
    
    if (vision_meas) {
      updateWithVision(vision_meas.value());
    }
    
    // Publish fused state estimate
    publishFusedState();
  }

 private:
  void initializeIMUModel() {
    // IMU noise characteristics
    imu_params_.gyro_noise_density = 0.000152;      // rad/s/sqrt(Hz)
    imu_params_.gyro_random_walk = 0.0000194;      // rad/s^2/sqrt(Hz)
    imu_params_.accel_noise_density = 0.004;        // m/s^2/sqrt(Hz)
    imu_params_.accel_random_walk = 0.006;          // m/s^3/sqrt(Hz)
    
    // IMU bias characteristics
    imu_params_.gyro_bias_correlation_time = 1e4;   // seconds
    imu_params_.accel_bias_correlation_time = 1e3;  // seconds
  }
  
  void initializeForceTorqueModel() {
    // Force/torque sensor characteristics
    ft_params_.force_resolution = 0.1;    // 0.1 N
    ft_params_.torque_resolution = 0.01;  // 0.01 Nm
    ft_params_.natural_frequency = 200.0; // Hz
    ft_params_.damping_ratio = 0.7;
  }
  
  void setupStateEstimator() {
    // Set up extended Kalman filter for humanoid state estimation
    // State vector: [pos, vel, ori, omega, acc_bias, gyro_bias]
    state_dim_ = 18;  // 3 pos + 3 vel + 4 ori + 3 omega + 3 accel_bias + 2 gyro_bias
    measurement_dim_ = 12;  // 6 IMU + 6 FT (simplified)
    
    // Initialize covariance matrix
    state_covariance_ = MatrixXd::Identity(state_dim_, state_dim_) * 1.0;
    process_noise_ = MatrixXd::Identity(state_dim_, state_dim_);
    measurement_noise_ = MatrixXd::Identity(measurement_dim_, measurement_dim_);
    
    // Initialize state
    state_estimate_ = VectorXd::Zero(state_dim_);
    state_estimate_.segment<4>(6) = Vector4d(0, 0, 0, 1);  // Identity quaternion
  }
  
  void updateWithIMU(const ImuMeasurement& meas) {
    // Prediction step with IMU data
    double dt = meas.timestamp - last_imu_time_;
    last_imu_time_ = meas.timestamp;
    
    // Update process model with IMU measurements
    predictState(meas.linear_acceleration, meas.angular_velocity, dt);
    
    // Update covariance
    predictCovariance(dt);
    
    // Construct measurement vector [linear_acc, angular_vel]
    Vector6d measurement;
    measurement << meas.linear_acceleration, meas.angular_velocity;
    
    // Measurement matrix (simplified)
    MatrixXd H = MatrixXd::Zero(6, state_dim_);
    H.block<3,3>(0, 9) = Matrix3d::Identity();  // Angular velocity measurement
    // Linear acceleration measurement would involve rotation matrix
    
    // Innovation
    Vector6d innovation = measurement - H * state_estimate_;
    
    // Innovation covariance
    Matrix6d innovation_cov = H * state_covariance_ * H.transpose() + 
                              Matrix6d::Identity() * imu_measurement_noise_;
    
    // Kalman gain
    MatrixXd kalman_gain = state_covariance_ * H.transpose() * 
                           innovation_cov.inverse();
    
    // Update state
    state_estimate_ += kalman_gain * innovation;
    
    // Update covariance
    state_covariance_ = (MatrixXd::Identity(state_dim_, state_dim_) - 
                        kalman_gain * H) * state_covariance_;
  }
  
  void updateWithForceTorque(const FtMeasurement& meas) {
    // Use force/torque measurements to detect ground contact
    double total_force = meas.wrench.force.norm();
    
    if (total_force > contact_threshold_) {
      // Confirmed ground contact
      // This can be used to correct position/velocity estimates
      correctWithContact(meas.contact_point, meas.wrench);
    }
  }
  
  void updateWithVision(const VisionMeasurement& meas) {
    // Vision provides absolute position measurements
    // Very valuable for correcting drift in IMU-only estimation
    
    if (meas.has_valid_pose) {
      // Absolute position measurement
      Vector3d vision_pos = meas.pose.translation();
      
      // Measurement matrix for position
      MatrixXd H = MatrixXd::Zero(3, state_dim_);
      H.block<3,3>(0, 0) = Matrix3d::Identity();  // Position measurement
      
      Vector3d measurement = vision_pos;
      Vector3d innovation = measurement - H * state_estimate_.head(3);
      
      Matrix3d innovation_cov = H.block<3,3>(0,0) * 
                               state_covariance_.block<3,3>(0,0) * 
                               H.block<3,3>(0,0).transpose() + 
                               Matrix3d::Identity() * vision_position_noise_;
      
      MatrixXd kalman_gain = state_covariance_.block<18,3>(0,0) * 
                             H.transpose() * innovation_cov.inverse();
      
      state_estimate_ += kalman_gain * innovation;
      state_covariance_ = (MatrixXd::Identity(state_dim_, state_dim_) - 
                          kalman_gain * H) * state_covariance_;
    }
  }
  
  void predictState(const Vector3d& accel, const Vector3d& omega, double dt) {
    // Simplified prediction model
    // In practice, this would use full rigid body dynamics
    
    // Update position based on velocity
    state_estimate_.segment<3>(0) += state_estimate_.segment<3>(3) * dt;
    
    // Update velocity based on acceleration
    state_estimate_.segment<3>(3) += accel * dt;
    
    // Update orientation based on angular velocity
    Vector4d quat = state_estimate_.segment<4>(6);
    Vector4d quat_dot = 0.5 * omegaToQuaternionMatrix(omega) * quat;
    state_estimate_.segment<4>(6) += quat_dot * dt;
    
    // Normalize quaternion
    state_estimate_.segment<4>(6).normalize();
  }
  
  void publishFusedState() {
    // Convert internal state to output format
    auto proto = tx_fused_state().initProto();
    
    // Position
    auto pos = proto.mutable_position();
    pos->set_x(state_estimate_[0]);
    pos->set_y(state_estimate_[1]);
    pos->set_z(state_estimate_[2]);
    
    // Velocity
    auto vel = proto.mutable_velocity();
    vel->set_x(state_estimate_[3]);
    vel->set_y(state_estimate_[4]);
    vel->set_z(state_estimate_[5]);
    
    // Orientation (quaternion)
    auto quat = proto.mutable_orientation();
    quat->set_w(state_estimate_[6]);
    quat->set_x(state_estimate_[7]);
    quat->set_y(state_estimate_[8]);
    quat->set_z(state_estimate_[9]);
    
    // Angular velocity
    auto omega = proto.mutable_angular_velocity();
    omega->set_x(state_estimate_[10]);
    omega->set_y(state_estimate_[11]);
    omega->set_z(state_estimate_[12]);
    
    tx_fused_state().publish();
  }

  // Sensor parameters
  struct IMUParams {
    double gyro_noise_density;
    double gyro_random_walk;
    double accel_noise_density;
    double accel_random_walk;
    double gyro_bias_correlation_time;
    double accel_bias_correlation_time;
  } imu_params_;
  
  struct FTParams {
    double force_resolution;
    double torque_resolution;
    double natural_frequency;
    double damping_ratio;
  } ft_params_;
  
  // State estimation
  int state_dim_;
  int measurement_dim_;
  VectorXd state_estimate_;
  MatrixXd state_covariance_;
  MatrixXd process_noise_;
  MatrixXd measurement_noise_;
  
  // Sensor fusion parameters
  double contact_threshold_ = 50.0;  // Newtons
  double imu_measurement_noise_ = 0.01;
  double vision_position_noise_ = 0.001;
  
  // Timing
  double last_imu_time_ = 0.0;
  double control_dt_ = 0.001;
  
  // Communication channels
  ISAAC_PROTO_RX(ImuMeasurementProto, imu_measurements);
  ISAAC_PROTO_RX(FtMeasurementProto, force_torque_measurements);
  ISAAC_PROTO_RX(JointEncoderProto, joint_encoder_measurements);
  ISAAC_PROTO_RX(VisionMeasurementProto, vision_measurements);
  ISAAC_PROTO_TX(FusedStateProto, fused_state);
};

}  // namespace fusion
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::fusion::LocomotionSensorFusion);
```

## Walking Performance Evaluation

### Metrics for Locomotion Quality
Evaluating the quality of bipedal locomotion:

```cpp
#include "engine/alice/alice.hpp"
#include "engine/gems/evaluation/locomotion_metrics.hpp"

namespace isaac {
namespace evaluation {

class WalkingPerformanceEvaluator : public Codelet {
 public:
  void start() override {
    // Initialize evaluation metrics
    initializeMetrics();
    
    // Set up data collection
    setupDataCollection();
    
    tickPeriodically(std::chrono::seconds(1));  // Evaluate every second
  }

  void tick() override {
    // Collect current performance data
    collectPerformanceData();
    
    // Calculate metrics
    calculateMetrics();
    
    // Log results
    logResults();
    
    // Check for performance degradation
    checkPerformanceDegradation();
  }

 private:
  void initializeMetrics() {
    // Define metrics to track
    metrics_["energy_efficiency"] = 0.0;
    metrics_["balance_stability"] = 0.0;
    metrics_["walking_smoothness"] = 0.0;
    metrics_["step_consistency"] = 0.0;
    metrics_["disturbance_recovery"] = 0.0;
    metrics_["walking_speed"] = 0.0;
  }
  
  void collectPerformanceData() {
    // Get current state
    auto state = rx_robot_state().popLatest();
    auto imu = rx_imu_data().popLatest();
    auto ft = rx_force_torque_sensors().popLatest();
    
    if (state && imu && ft) {
      // Store for metric calculations
      recent_states_.push_back(state.value());
      recent_imu_.push_back(imu.value());
      recent_ft_.push_back(ft.value());
      
      // Keep only recent data (last 10 seconds)
      while (recent_states_.size() > 1000) {  // Assuming 100 Hz sampling
        recent_states_.pop_front();
        recent_imu_.pop_front();
        recent_ft_.pop_front();
      }
    }
  }
  
  void calculateMetrics() {
    if (recent_states_.size() < 100) return;  // Need sufficient data
    
    // Calculate energy efficiency (cost of transport)
    metrics_["energy_efficiency"] = calculateCostOfTransport();
    
    // Calculate balance stability (ZMP margin, COM variance)
    metrics_["balance_stability"] = calculateBalanceStability();
    
    // Calculate walking smoothness (jerk minimization)
    metrics_["walking_smoothness"] = calculateWalkingSmoothness();
    
    // Calculate step consistency (regularity of steps)
    metrics_["step_consistency"] = calculateStepConsistency();
    
    // Calculate walking speed
    metrics_["walking_speed"] = calculateWalkingSpeed();
    
    // Disturbance recovery would require external disturbances
    // This is calculated when disturbances are applied
  }
  
  double calculateCostOfTransport() {
    // Cost of transport = metabolic power / (body_weight * velocity)
    // For robots: Power / (mass * gravity * velocity)
    
    // Calculate instantaneous power consumption
    double total_power = 0.0;
    auto last_state = recent_states_.back();
    auto last_joints = last_state.joint_states();
    
    for (int i = 0; i < last_joints.size(); i++) {
      double torque = last_joints[i].effort();
      double velocity = last_joints[i].velocity();
      total_power += abs(torque * velocity);
    }
    
    // Calculate average velocity
    double avg_velocity = calculateAverageVelocity();
    
    if (avg_velocity > 0.01) {  // Avoid division by zero
      double cost_of_transport = total_power / (body_mass_ * gravity_ * avg_velocity);
      return 1.0 / (cost_of_transport + 0.1);  // Invert so higher is better
    }
    
    return 0.0;
  }
  
  double calculateBalanceStability() {
    // Measure how close ZMP stays to center of support
    double stability_score = 0.0;
    int count = 0;
    
    for (const auto& state : recent_states_) {
      Vector2d zmp = calculateZMPFromState(state);
      Vector2d support_center = calculateSupportCenter(state);
      double distance_to_center = (zmp - support_center).norm();
      
      // Distance to edge of support polygon
      double distance_to_edge = calculateDistanceToSupportEdge(zmp, state);
      
      // Score based on margin from edge
      stability_score += distance_to_edge;
      count++;
    }
    
    if (count > 0) {
      return stability_score / count;
    }
    
    return 0.0;
  }
  
  double calculateWalkingSmoothness() {
    // Measure smoothness by calculating jerk (derivative of acceleration)
    if (recent_states_.size() < 3) return 0.0;
    
    double total_jerk = 0.0;
    int count = 0;
    
    auto it = recent_states_.begin();
    auto prev_it = it++;  // First element
    auto curr_it = it++;  // Second element
    
    while (it != recent_states_.end()) {
      auto next_state = *it;
      auto curr_state = *curr_it;
      auto prev_state = *prev_it;
      
      // Calculate acceleration for current state
      Vector3d curr_vel = getStateVelocity(curr_state);
      Vector3d prev_vel = getStateVelocity(prev_state);
      Vector3d next_vel = getStateVelocity(next_state);
      
      Vector3d curr_acc = (curr_vel - prev_vel) / control_dt_;
      Vector3d next_acc = (next_vel - curr_vel) / control_dt_;
      
      Vector3d jerk = (next_acc - curr_acc) / control_dt_;
      total_jerk += jerk.norm();
      
      prev_it = curr_it;
      curr_it = it++;
      count++;
    }
    
    if (count > 0) {
      // Lower jerk is better, so invert
      return 1.0 / (total_jerk / count + 0.001);
    }
    
    return 0.0;
  }
  
  double calculateStepConsistency() {
    // Measure consistency of step timing and positioning
    if (step_events_.size() < 2) return 0.0;
    
    // Calculate step time variability
    double total_time_diff = 0.0;
    for (size_t i = 1; i < step_events_.size(); i++) {
      double time_diff = step_events_[i].time - step_events_[i-1].time;
      total_time_diff += abs(time_diff - nominal_step_time_);
    }
    
    double time_variability = total_time_diff / (step_events_.size() - 1);
    
    // Calculate step position variability
    double total_pos_diff = 0.0;
    for (const auto& step : step_events_) {
      double pos_error = calculateStepPositionError(step);
      total_pos_diff += pos_error;
    }
    
    double pos_variability = total_pos_diff / step_events_.size();
    
    // Combine metrics (lower variability is better, so invert)
    return 1.0 / (time_variability + pos_variability + 0.001);
  }
  
  void logResults() {
    LOG_INFO("Walking Performance Metrics:");
    LOG_INFO("  Energy Efficiency: %.3f", metrics_["energy_efficiency"]);
    LOG_INFO("  Balance Stability: %.3f", metrics_["balance_stability"]);
    LOG_INFO("  Walking Smoothness: %.3f", metrics_["walking_smoothness"]);
    LOG_INFO("  Step Consistency: %.3f", metrics_["step_consistency"]);
    LOG_INFO("  Walking Speed: %.3f m/s", metrics_["walking_speed"]);
  }
  
  void checkPerformanceDegradation() {
    // Compare current performance to baseline
    if (metrics_["balance_stability"] < stability_threshold_) {
      LOG_WARN("Balance stability degrading, consider adjusting controller");
    }
    
    if (metrics_["walking_smoothness"] < smoothness_threshold_) {
      LOG_WARN("Walking smoothness degrading, check for mechanical issues");
    }
  }

  // Metrics storage
  std::map<std::string, double> metrics_;
  
  // Data collections
  std::deque<RobotState> recent_states_;
  std::deque<ImuData> recent_imu_;
  std::deque<ForceTorqueData> recent_ft_;
  
  // Step tracking
  std::vector<StepEvent> step_events_;
  double nominal_step_time_ = 1.0;
  
  // Performance thresholds
  double stability_threshold_ = 0.05;
  double smoothness_threshold_ = 0.1;
  
  // Constants
  double body_mass_ = 75.0;  // kg
  double gravity_ = 9.81;   // m/s^2
  double control_dt_ = 0.01; // 100 Hz
  
  // Communication channels
  ISAAC_PROTO_RX(RobotStateProto, robot_state);
  ISAAC_PROTO_RX(ImuDataProto, imu_data);
  ISAAC_PROTO_RX(ForceTorqueProto, force_torque_sensors);
  ISAAC_PROTO_TX(PerformanceMetricsProto, performance_metrics);
};

}  // namespace evaluation
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::evaluation::WalkingPerformanceEvaluator);
```

## Best Practices

### Controller Design
1. **Hierarchical Control**: Implement controllers at different time scales
2. **Robustness**: Design controllers that work across different terrains
3. **Adaptability**: Allow parameters to adapt to different walking speeds
4. **Safety**: Include safety limits and fallback behaviors

### Balance Strategy Selection
1. **Minimal Intervention**: Use the least disruptive strategy first
2. **Smooth Transitions**: Ensure smooth transitions between strategies
3. **Predictive Control**: Anticipate balance losses before they occur
4. **Recovery Planning**: Plan recovery actions before they're needed

### Performance Optimization
1. **Energy Efficiency**: Optimize for minimal energy consumption
2. **Stability Margins**: Maintain adequate stability margins
3. **Smooth Transitions**: Avoid abrupt changes in control
4. **Real-time Capability**: Ensure controllers run in real-time

## Summary

Bipedal locomotion and balance control are among the most challenging aspects of humanoid robotics. Success requires integrating multiple control strategies, sensory feedback systems, and predictive control methods. The key to stable walking lies in understanding the dynamic nature of bipedal gait and implementing appropriate balance recovery mechanisms.

## Exercises

1. Implement a simple PD controller for ankle-based balance.
2. Design a stepping controller for balance recovery.
3. Create a sensor fusion algorithm combining IMU and force/torque data.
4. Evaluate the stability of a walking gait using ZMP analysis.