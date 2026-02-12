---
sidebar_position: 1
---

# Week 11: Humanoid Kinematics & Dynamics

## Learning Objectives

By the end of this week, you will be able to:
- Understand the kinematic structure of humanoid robots
- Implement forward and inverse kinematics for humanoid systems
- Model the dynamics of multi-body systems
- Apply Lagrangian mechanics to derive equations of motion
- Implement control strategies for humanoid balance
- Understand the challenges of humanoid locomotion
- Analyze the stability of humanoid systems

## Introduction to Humanoid Robotics

### What Makes Humanoids Unique?
Humanoid robots are designed to mimic human form and capabilities, featuring:
- **Bipedal locomotion**: Two-legged walking similar to humans
- **Anthropomorphic structure**: Human-like proportions and degrees of freedom
- **Upper limb dexterity**: Arms and hands for manipulation tasks
- **Social interaction**: Designed for human-compatible environments

### Key Challenges in Humanoid Robotics
- **Balance Control**: Maintaining stability with narrow support base
- **Dynamic Locomotion**: Achieving stable walking and running
- **Whole-Body Coordination**: Coordinating multiple limbs for tasks
- **Real-time Control**: Meeting strict timing constraints for stability

## Humanoid Kinematics

### Kinematic Structure
Humanoid robots typically have a kinematic structure similar to humans:

```
        Head
         |
      Torso
      /     \
   Left    Right
  Arm      Arm
   |        |
  Hand    Hand
   |        |
  Left    Right
 Foot    Foot
```

### Denavit-Hartenberg Convention for Humanoids
The DH convention can be applied to humanoid limbs:

```cpp
#include "engine/alice/alice.hpp"
#include "engine/gems/kinematics/dh_parameters.hpp"

namespace isaac {
namespace humanoid {

class HumanoidKinematics : public Codelet {
 public:
  void start() override {
    // Initialize kinematic models for each limb
    initializeLeftArmKinematics();
    initializeRightArmKinematics();
    initializeLeftLegKinematics();
    initializeRightLegKinematics();
    initializeTorsoKinematics();
    
    // Set up whole-body kinematic chain
    setupWholeBodyKinematics();
  }

  void tick() override {
    // Update kinematic solution based on joint angles
    if (rx_joint_angles().available()) {
      auto joint_angles = rx_joint_angles().popLatest().value();
      
      // Compute forward kinematics
      computeForwardKinematics(joint_angles);
      
      // Publish end-effector poses
      publishEndEffectorPoses();
    }
  }

 private:
  void initializeLeftArmKinematics() {
    // Define DH parameters for left arm
    dh_params_left_arm_ = {
      {0.0, M_PI/2, 0.0, 0.1},    // Shoulder yaw
      {0.0, -M_PI/2, 0.15, 0.0},  // Shoulder pitch
      {0.0, M_PI/2, 0.0, 0.25},   // Shoulder roll
      {0.0, -M_PI/2, 0.0, 0.0},   // Elbow pitch
      {0.0, M_PI/2, 0.2, 0.0},    // Elbow yaw
      {0.0, 0.0, 0.0, 0.15}       // Wrist pitch
    };
    
    left_arm_chain_ = DHChain(dh_params_left_arm_);
  }
  
  void initializeRightArmKinematics() {
    // Similar to left arm but mirrored
    dh_params_right_arm_ = {
      {0.0, M_PI/2, 0.0, 0.1},    // Shoulder yaw
      {0.0, M_PI/2, 0.15, 0.0},   // Shoulder pitch (opposite direction)
      {0.0, -M_PI/2, 0.0, 0.25},  // Shoulder roll
      {0.0, M_PI/2, 0.0, 0.0},    // Elbow pitch
      {0.0, -M_PI/2, 0.2, 0.0},   // Elbow yaw
      {0.0, 0.0, 0.0, 0.15}       // Wrist pitch
    };
    
    right_arm_chain_ = DHChain(dh_params_right_arm_);
  }
  
  void initializeLeftLegKinematics() {
    // Define DH parameters for left leg
    dh_params_left_leg_ = {
      {0.0, -M_PI/2, 0.0, 0.05},  // Hip yaw
      {0.0, M_PI/2, 0.05, 0.0},   // Hip roll
      {0.0, -M_PI/2, 0.0, 0.0},   // Hip pitch
      {0.0, 0.0, -0.35, 0.0},     // Knee pitch
      {0.0, 0.0, -0.35, 0.0},     // Ankle pitch
      {0.0, M_PI/2, 0.0, 0.0}     // Ankle roll
    };
    
    left_leg_chain_ = DHChain(dh_params_left_leg_);
  }
  
  void initializeRightLegKinematics() {
    // Similar to left leg but mirrored
    dh_params_right_leg_ = {
      {0.0, M_PI/2, 0.0, 0.05},   // Hip yaw
      {0.0, -M_PI/2, 0.05, 0.0},  // Hip roll (opposite direction)
      {0.0, M_PI/2, 0.0, 0.0},    // Hip pitch
      {0.0, 0.0, -0.35, 0.0},     // Knee pitch
      {0.0, 0.0, -0.35, 0.0},     // Ankle pitch
      {0.0, -M_PI/2, 0.0, 0.0}    // Ankle roll
    };
    
    right_leg_chain_ = DHChain(dh_params_right_leg_);
  }
  
  void setupWholeBodyKinematics() {
    // Define base coordinate frames
    torso_frame_ = Pose3d::Identity();  // Torso is the base
    
    // Define offsets from torso to limb bases
    left_shoulder_offset_ = Pose3d(Vector3d(0.0, 0.15, 0.1), Quaterniond::Identity());
    right_shoulder_offset_ = Pose3d(Vector3d(0.0, -0.15, 0.1), Quaterniond::Identity());
    left_hip_offset_ = Pose3d(Vector3d(0.0, 0.05, -0.05), Quaterniond::Identity());
    right_hip_offset_ = Pose3d(Vector3d(0.0, -0.05, -0.05), Quaterniond::Identity());
  }
  
  void computeForwardKinematics(const JointAngles& joint_angles) {
    // Compute forward kinematics for each limb
    
    // Left arm
    std::vector<double> left_arm_joints = extractLeftArmJoints(joint_angles);
    left_hand_pose_ = left_shoulder_offset_ * left_arm_chain_.forwardKinematics(left_arm_joints);
    
    // Right arm
    std::vector<double> right_arm_joints = extractRightArmJoints(joint_angles);
    right_hand_pose_ = right_shoulder_offset_ * right_arm_chain_.forwardKinematics(right_arm_joints);
    
    // Left leg
    std::vector<double> left_leg_joints = extractLeftLegJoints(joint_angles);
    left_foot_pose_ = left_hip_offset_ * left_leg_chain_.forwardKinematics(left_leg_joints);
    
    // Right leg
    std::vector<double> right_leg_joints = extractRightLegJoints(joint_angles);
    right_foot_pose_ = right_hip_offset_ * right_leg_chain_.forwardKinematics(right_leg_joints);
    
    // Compute center of mass based on all links
    computeCenterOfMass(joint_angles);
  }
  
  void computeCenterOfMass(const JointAngles& joint_angles) {
    // Compute COM using link masses and positions
    Vector3d total_momentum = Vector3d::Zero();
    double total_mass = 0.0;
    
    // Iterate through all links and compute weighted position
    for (int i = 0; i < num_links_; i++) {
      Pose3d link_pose = computeLinkPose(i, joint_angles);
      double link_mass = link_masses_[i];
      
      Vector3d com_contribution = link_pose.translation * link_mass;
      total_momentum += com_contribution;
      total_mass += link_mass;
    }
    
    if (total_mass > 0.0) {
      center_of_mass_ = total_momentum / total_mass;
    }
  }
  
  void publishEndEffectorPoses() {
    // Publish computed poses
    auto left_hand_msg = tx_left_hand_pose().initProto();
    setPoseProto(left_hand_msg, left_hand_pose_);
    tx_left_hand_pose().publish();
    
    auto right_hand_msg = tx_right_hand_pose().initProto();
    setPoseProto(right_hand_msg, right_hand_pose_);
    tx_right_hand_pose().publish();
    
    auto left_foot_msg = tx_left_foot_pose().initProto();
    setPoseProto(left_foot_msg, left_foot_pose_);
    tx_left_foot_pose().publish();
    
    auto right_foot_msg = tx_right_foot_pose().initProto();
    setPoseProto(right_foot_msg, right_foot_pose_);
    tx_right_foot_pose().publish();
    
    // Publish center of mass
    auto com_msg = tx_com_pose().initProto();
    setPositionProto(com_msg.mutable_position(), center_of_mass_);
    tx_com_pose().publish();
  }

  // DH parameters for each limb
  std::vector<DHParameter> dh_params_left_arm_;
  std::vector<DHParameter> dh_params_right_arm_;
  std::vector<DHParameter> dh_params_left_leg_;
  std::vector<DHParameter> dh_params_right_leg_;
  
  // Kinematic chains
  DHChain left_arm_chain_;
  DHChain right_arm_chain_;
  DHChain left_leg_chain_;
  DHChain right_leg_chain_;
  
  // Coordinate frame offsets
  Pose3d torso_frame_;
  Pose3d left_shoulder_offset_;
  Pose3d right_shoulder_offset_;
  Pose3d left_hip_offset_;
  Pose3d right_hip_offset_;
  
  // End-effector poses
  Pose3d left_hand_pose_;
  Pose3d right_hand_pose_;
  Pose3d left_foot_pose_;
  Pose3d right_foot_pose_;
  Vector3d center_of_mass_;
  
  // Link properties
  int num_links_ = 30;  // Example number
  std::vector<double> link_masses_;
  
  // Communication channels
  ISAAC_PROTO_RX(JointAnglesProto, joint_angles);
  ISAAC_PROTO_TX(Pose3dProto, left_hand_pose);
  ISAAC_PROTO_TX(Pose3dProto, right_hand_pose);
  ISAAC_PROTO_TX(Pose3dProto, left_foot_pose);
  ISAAC_PROTO_TX(Pose3dProto, right_foot_pose);
  ISAAC_PROTO_TX(Pose3dProto, com_pose);
};

}  // namespace humanoid
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::humanoid::HumanoidKinematics);
```

## Forward Kinematics

### Mathematical Representation
Forward kinematics computes the end-effector position given joint angles:

```cpp
// Implementation of forward kinematics for a serial chain
class ForwardKinematics {
 public:
  ForwardKinematics(const std::vector<DHParameter>& dh_params) 
      : dh_params_(dh_params) {}
  
  Pose3d compute(const std::vector<double>& joint_angles) {
    if (joint_angles.size() != dh_params_.size()) {
      throw std::invalid_argument("Joint angle count mismatch");
    }
    
    Pose3d cumulative_transform = Pose3d::Identity();
    
    for (size_t i = 0; i < dh_params_.size(); ++i) {
      // Create transform for this joint using DH parameters
      Pose3d joint_transform = dhTransform(dh_params_[i], joint_angles[i]);
      
      // Apply transform
      cumulative_transform = cumulative_transform * joint_transform;
    }
    
    return cumulative_transform;
  }

 private:
  Pose3d dhTransform(const DHParameter& params, double theta) {
    // DH transformation matrix
    double ct = cos(theta + params.theta_offset);
    double st = sin(theta + params.theta_offset);
    double ca = cos(params.alpha);
    double sa = sin(params.alpha);
    
    Matrix3d rotation;
    rotation << ct, -st * ca, st * sa,
                st, ct * ca, -ct * sa,
                0, sa, ca;
    
    Vector3d translation(params.a, -sa * params.d, ca * params.d);
    
    return Pose3d(translation, Quaterniond(rotation));
  }
  
  std::vector<DHParameter> dh_params_;
};
```

## Inverse Kinematics

### Jacobian-Based Approach
Inverse kinematics solves for joint angles given end-effector pose:

```cpp
#include "engine/alice/alice.hpp"
#include "engine/gems/kinematics/jacobian.hpp"

namespace isaac {
namespace humanoid {

class HumanoidInverseKinematics : public Codelet {
 public:
  void start() override {
    // Initialize IK solvers for each limb
    initializeLeftArmIK();
    initializeRightArmIK();
    initializeLeftLegIK();
    initializeRightLegIK();
    
    // Set up constraints
    setupConstraints();
    
    tickOnMessage(rx_target_pose());
  }

  void tick() override {
    if (rx_target_pose().available()) {
      auto target_msg = rx_target_pose().popLatest().value();
      
      // Determine which limb to control
      std::string limb = target_msg.limb();
      Pose3d target_pose = protoToPose3d(target_msg.target_pose());
      
      // Get current joint angles
      auto current_joints = rx_current_joints().popLatest().value();
      
      // Solve inverse kinematics
      std::vector<double> new_joints;
      bool success = false;
      
      if (limb == "left_arm") {
        success = solveLeftArmIK(target_pose, current_joints, new_joints);
      } else if (limb == "right_arm") {
        success = solveRightArmIK(target_pose, current_joints, new_joints);
      } else if (limb == "left_leg") {
        success = solveLeftLegIK(target_pose, current_joints, new_joints);
      } else if (limb == "right_leg") {
        success = solveRightLegIK(target_pose, current_joints, new_joints);
      }
      
      if (success) {
        // Publish new joint angles
        publishJointCommands(new_joints, limb);
      } else {
        LOG_WARN("IK solution not found for %s", limb.c_str());
      }
    }
  }

 private:
  bool solveLeftArmIK(const Pose3d& target_pose, 
                     const JointAngles& current_joints,
                     std::vector<double>& solution) {
    // Extract current left arm joint angles
    std::vector<double> current_arm_joints = extractLeftArmJoints(current_joints);
    
    // Compute initial end-effector pose
    Pose3d current_pose = left_arm_fk_.forwardKinematics(current_arm_joints);
    
    // Iterative IK solver using Jacobian transpose
    std::vector<double> joints = current_arm_joints;
    
    for (int iter = 0; iter < max_iterations_; iter++) {
      // Compute current end-effector pose
      Pose3d current_effector_pose = left_arm_fk_.forwardKinematics(joints);
      
      // Compute pose error
      Vector6d error = poseDifference(target_pose, current_effector_pose);
      
      // Check convergence
      if (error.norm() < position_tolerance_) {
        solution = joints;
        return true;
      }
      
      // Compute Jacobian
      MatrixXd jacobian = left_arm_jacobian_.compute(joints);
      
      // Compute joint updates using Jacobian transpose
      VectorXd delta_joints = jacobian.transpose() * error * learning_rate_;
      
      // Apply joint limits
      for (size_t i = 0; i < joints.size(); i++) {
        joints[i] += delta_joints[i];
        joints[i] = clamp(joints[i], joint_limits_min_[i], joint_limits_max_[i]);
      }
    }
    
    // If we didn't converge, return false
    return false;
  }
  
  Vector6d poseDifference(const Pose3d& target, const Pose3d& current) {
    // Compute difference in position and orientation
    Vector6d diff;
    
    // Position difference
    diff.segment<3>(0) = target.translation - current.translation;
    
    // Orientation difference (using logarithmic map)
    Quaterniond q_diff = target.rotation * current.rotation.inverse();
    Vector3d rot_vec = quaternionToRotationVector(q_diff);
    diff.segment<3>(3) = rot_vec;
    
    return diff;
  }
  
  MatrixXd computeJacobian(const std::vector<double>& joint_angles, int num_dofs) {
    // Compute analytical Jacobian for the manipulator
    MatrixXd jacobian = MatrixXd::Zero(6, num_dofs);
    
    // Get link poses
    std::vector<Pose3d> link_poses = getAllLinkPoses(joint_angles);
    Pose3d end_effector_pose = link_poses.back();
    
    // Compute Jacobian columns
    for (int i = 0; i < num_dofs; i++) {
      // For revolute joints
      Vector3d joint_axis = link_poses[i].rotation * Vector3d::UnitZ();  // Assuming z-axis rotation
      Vector3d joint_to_end_effector = end_effector_pose.translation - link_poses[i].translation;
      
      // Linear velocity component
      jacobian.block<3,1>(0, i) = joint_axis.cross(joint_to_end_effector);
      
      // Angular velocity component
      jacobian.block<3,1>(3, i) = joint_axis;
    }
    
    return jacobian;
  }
  
  void setupConstraints() {
    // Joint limits
    joint_limits_min_ = {-1.57, -2.0, -3.14, -2.5, -2.0, -2.0};  // Example limits
    joint_limits_max_ = {1.57, 1.57, 3.14, 2.5, 2.0, 2.0};
    
    // Velocity limits
    velocity_limits_ = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  }

  // IK solvers for each limb
  ForwardKinematics left_arm_fk_;
  ForwardKinematics right_arm_fk_;
  ForwardKinematics left_leg_fk_;
  ForwardKinematics right_leg_fk_;
  
  JacobianCalculator left_arm_jacobian_;
  JacobianCalculator right_arm_jacobian_;
  JacobianCalculator left_leg_jacobian_;
  JacobianCalculator right_leg_jacobian_;
  
  // Parameters
  int max_iterations_ = 100;
  double position_tolerance_ = 0.001;  // 1mm
  double orientation_tolerance_ = 0.01; // 0.01 rad
  double learning_rate_ = 0.5;
  
  // Joint limits
  std::vector<double> joint_limits_min_;
  std::vector<double> joint_limits_max_;
  std::vector<double> velocity_limits_;
  
  // Communication channels
  ISAAC_PROTO_RX(TargetPoseProto, target_pose);
  ISAAC_PROTO_RX(JointAnglesProto, current_joints);
  ISAAC_PROTO_TX(JointCommandsProto, joint_commands);
};

}  // namespace humanoid
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::humanoid::HumanoidInverseKinematics);
```

## Humanoid Dynamics

### Lagrangian Mechanics
The dynamics of a humanoid robot can be modeled using Lagrangian mechanics:

```cpp
#include "engine/alice/alice.hpp"
#include "engine/gems/dynamics/lagrangian_dynamics.hpp"

namespace isaac {
namespace humanoid {

class HumanoidDynamics : public Codelet {
 public:
  void start() override {
    // Initialize dynamic model
    initializeDynamicModel();
    
    // Set up control parameters
    setupControlParameters();
    
    tickPeriodically(std::chrono::milliseconds(1));  // 1000 Hz for dynamics
  }

  void tick() override {
    if (rx_joint_states().available()) {
      auto joint_states = rx_joint_states().popLatest().value();
      
      // Get current state
      auto joint_positions = joint_states.positions();
      auto joint_velocities = joint_states.velocities();
      
      // Compute dynamic terms
      MatrixXd mass_matrix = computeMassMatrix(joint_positions);
      VectorXd coriolis_centrifugal = computeCoriolisCentrifugal(joint_positions, joint_velocities);
      VectorXd gravity_term = computeGravityTerm(joint_positions);
      
      // Compute required torques for desired motion
      VectorXd desired_acc = computeDesiredAccelerations();
      VectorXd required_torques = mass_matrix * desired_acc + coriolis_centrifugal + gravity_term;
      
      // Apply external forces (contacts, etc.)
      VectorXd external_forces = computeExternalForces();
      required_torques += external_forces;
      
      // Publish required torques
      publishRequiredTorques(required_torques);
      
      // Also compute and publish dynamic properties
      publishDynamicProperties(mass_matrix, coriolis_centrifugal, gravity_term);
    }
  }

 private:
  void initializeDynamicModel() {
    // Define link properties (mass, inertia, COM)
    links_.resize(num_joints_);
    
    for (int i = 0; i < num_joints_; i++) {
      links_[i].mass = link_masses_[i];
      links_[i].inertia = link_inertias_[i];  // 3x3 inertia matrix
      links_[i].com = link_com_offsets_[i];   // COM offset from joint
    }
    
    // Define kinematic structure (DH parameters or similar)
    kinematic_structure_ = buildKinematicTree();
  }
  
  MatrixXd computeMassMatrix(const std::vector<double>& q) {
    // Compute the mass/inertia matrix using recursive algorithms
    // This is a simplified version - in practice, use composite rigid body algorithm
    
    int n = q.size();
    MatrixXd M = MatrixXd::Zero(n, n);
    
    // For each joint, compute its contribution to the mass matrix
    for (int i = 0; i < n; i++) {
      // Compute position and orientation of each link
      std::vector<Pose3d> link_poses = computeLinkPoses(q);
      
      for (int j = 0; j < n; j++) {
        // Compute the effect of joint j's motion on joint i
        M(i, j) = computeInertialCoupling(link_poses, i, j);
      }
    }
    
    return M;
  }
  
  VectorXd computeCoriolisCentrifugal(const std::vector<double>& q, 
                                   const std::vector<double>& qdot) {
    // Compute Coriolis and centrifugal forces
    int n = q.size();
    VectorXd C = VectorXd::Zero(n);
    
    // Compute Christoffel symbols and apply them
    MatrixXd M = computeMassMatrix(q);
    
    // Use the identity: C(q, q̇) = ½∑(∂M/∂q_k)q̇_i*q̇_j for all i,j,k
    for (int k = 0; k < n; k++) {
      MatrixXd dM_dqk = computeMassMatrixDerivative(M, q, k);
      
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          C[k] += 0.5 * (dM_dqk(i, j) - 
                         dM_dqk(k, i) - 
                         dM_dqk(k, j)) * qdot[i] * qdot[j];
        }
      }
    }
    
    return C;
  }
  
  VectorXd computeGravityTerm(const std::vector<double>& q) {
    // Compute gravitational forces
    int n = q.size();
    VectorXd G = VectorXd::Zero(n);
    
    // Compute potential energy gradient
    std::vector<Pose3d> link_poses = computeLinkPoses(q);
    
    for (int i = 0; i < n; i++) {
      // Compute the effect of gravity on each link
      Vector3d gravity_effect = computeGravityEffectOnLink(link_poses[i], i);
      
      // Project onto joint axis
      Vector3d joint_axis = computeJointAxis(i, q);
      G[i] = gravity_effect.dot(joint_axis);
    }
    
    return G;
  }
  
  VectorXd computeDesiredAccelerations() {
    // Compute desired joint accelerations based on control law
    // This would come from a higher-level controller
    
    // For example, using PD control:
    // τ = M(q)q̈ᵈ + C(q,q̇)q̇ᵈ + G(q) + Kp(qᵈ-q) + Kd(q̇ᵈ-q̇)
    
    VectorXd q_desired_ddot = VectorXd::Zero(num_joints_);
    
    // This would be computed by a trajectory generator
    // For now, return zeros
    return q_desired_ddot;
  }
  
  VectorXd computeExternalForces() {
    // Compute forces due to contacts, constraints, etc.
    VectorXd external_forces = VectorXd::Zero(num_joints_);
    
    // Check for ground contacts
    auto ground_contacts = detectGroundContacts();
    
    for (const auto& contact : ground_contacts) {
      // Apply contact forces to the appropriate joints
      applyContactForce(external_forces, contact);
    }
    
    return external_forces;
  }
  
  void setupControlParameters() {
    // Define control gains
    kp_ = VectorXd::Constant(num_joints_, 100.0);  // Position gains
    kd_ = VectorXd::Constant(num_joints_, 20.0);   // Velocity gains
  }

  // Dynamic model components
  std::vector<LinkProperties> links_;
  KinematicTree kinematic_structure_;
  
  // Control parameters
  VectorXd kp_;
  VectorXd kd_;
  
  // Constants
  int num_joints_ = 28;  // Example: 6 DOF per leg/arm, 4 DOF torso, 2 DOF head
  
  // Link properties
  std::vector<double> link_masses_ = {/* initialized with actual values */};
  std::vector<Matrix3d> link_inertias_ = {/* initialized with actual values */};
  std::vector<Vector3d> link_com_offsets_ = {/* initialized with actual values */};
  
  // Communication channels
  ISAAC_PROTO_RX(JointStatesProto, joint_states);
  ISAAC_PROTO_TX(JointTorquesProto, joint_torques);
  ISAAC_PROTO_TX(DynamicPropertiesProto, dynamic_properties);
};

}  // namespace humanoid
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::humanoid::HumanoidDynamics);
```

## Balance Control

### Zero Moment Point (ZMP)
ZMP is a critical concept for humanoid balance:

```cpp
#include "engine/alice/alice.hpp"
#include "engine/gems/control/zmp_calculator.hpp"

namespace isaac {
namespace humanoid {

class BalanceController : public Codelet {
 public:
  void start() override {
    // Initialize ZMP calculation
    initializeZMPCalculation();
    
    // Set up balance control parameters
    setupBalanceControl();
    
    tickPeriodically(std::chrono::milliseconds(10));  // 100 Hz control
  }

  void tick() override {
    // Get current state
    auto joint_states = rx_joint_states().popLatest().value();
    auto imu_data = rx_imu_data().popLatest().value();
    
    // Compute center of mass
    auto com = computeCenterOfMass(joint_states.positions());
    
    // Compute ZMP
    auto zmp = computeZMP(com, joint_states, imu_data);
    
    // Compute desired ZMP based on walking pattern
    auto desired_zmp = computeDesiredZMP();
    
    // Compute balance correction
    auto balance_correction = computeBalanceCorrection(zmp, desired_zmp);
    
    // Apply balance control
    applyBalanceControl(balance_correction, joint_states);
    
    // Publish ZMP for monitoring
    publishZMPInfo(zmp, desired_zmp);
  }

 private:
  void initializeZMPCalculation() {
    // Set up parameters for ZMP calculation
    gravity_ = 9.81;
    com_height_ = 0.8;  // Average COM height for humanoid
    
    // Define support polygon based on foot geometry
    defineSupportPolygon();
  }
  
  Vector2d computeZMP(const Vector3d& com, 
                     const JointStates& joint_states,
                     const ImuData& imu_data) {
    // ZMP calculation using the formula:
    // ZMP_x = com_x - (com_z - z_support) * ẍ_com / g
    // ZMP_y = com_y - (com_z - z_support) * ÿ_com / g
    
    // Compute COM acceleration (simplified)
    Vector3d com_acceleration = estimateCOMAcceleration(com, joint_states);
    
    // Calculate ZMP
    Vector2d zmp;
    zmp.x() = com.x() - ((com.z() - support_surface_z_) * com_acceleration.x()) / gravity_;
    zmp.y() = com.y() - ((com.z() - support_surface_z_) * com_acceleration.y()) / gravity_;
    
    return zmp;
  }
  
  Vector2d computeDesiredZMP() {
    // Generate desired ZMP trajectory based on walking pattern
    // This would typically come from a walking pattern generator
    
    // For standing, desired ZMP is usually at the center of the support polygon
    if (standing_mode_) {
      return support_polygon_center_;
    }
    
    // For walking, interpolate along the desired path
    return computeWalkingZMP();
  }
  
  Vector3d computeBalanceCorrection(const Vector2d& actual_zmp, 
                                  const Vector2d& desired_zmp) {
    // Compute error
    Vector2d zmp_error = desired_zmp - actual_zmp;
    
    // Apply feedback control
    Vector3d correction;
    correction.x() = kp_zmp_pos_ * zmp_error.x();
    correction.y() = kp_zmp_pos_ * zmp_error.y();
    correction.z() = 0;  // No vertical correction needed
    
    // Apply saturation limits
    correction.x() = clamp(correction.x(), -max_correction_, max_correction_);
    correction.y() = clamp(correction.y(), -max_correction_, max_correction_);
    
    return correction;
  }
  
  void applyBalanceControl(const Vector3d& correction, 
                         const JointStates& current_states) {
    // Apply balance correction through ankle strategy
    if (use_ankle_strategy_) {
      applyAnkleStrategy(correction);
    }
    
    // Apply hip strategy if needed
    if (use_hip_strategy_) {
      applyHipStrategy(correction);
    }
    
    // Apply arm strategy for additional balance
    if (use_arm_strategy_) {
      applyArmStrategy(correction);
    }
  }
  
  void applyAnkleStrategy(const Vector3d& correction) {
    // Adjust ankle angles to shift COM
    double ankle_roll_correction = correction.y() * ankle_gain_;
    double ankle_pitch_correction = correction.x() * ankle_gain_;
    
    // Apply corrections to ankle joints
    ankle_roll_command_ += ankle_roll_correction;
    ankle_pitch_command_ += ankle_pitch_correction;
    
    // Publish ankle commands
    publishAnkleCommands(ankle_roll_command_, ankle_pitch_command_);
  }
  
  void defineSupportPolygon() {
    // Define polygon based on foot geometry
    // For a rectangular foot: 4 corners
    support_polygon_.clear();
    support_polygon_.push_back(Vector2d(foot_length_/2, foot_width_/2));    // front-left
    support_polygon_.push_back(Vector2d(foot_length_/2, -foot_width_/2));   // front-right
    support_polygon_.push_back(Vector2d(-foot_length_/2, -foot_width_/2));  // back-right
    support_polygon_.push_back(Vector2d(-foot_length_/2, foot_width_/2));   // back-left
    
    // Compute centroid
    support_polygon_center_ = Vector2d::Zero();
    for (const auto& vertex : support_polygon_) {
      support_polygon_center_ += vertex;
    }
    support_polygon_center_ /= support_polygon_.size();
  }
  
  bool isZMPStable(const Vector2d& zmp) {
    // Check if ZMP is within support polygon
    return isPointInPolygon(zmp, support_polygon_);
  }
  
  void setupBalanceControl() {
    // Balance control parameters
    kp_zmp_pos_ = 50.0;      // ZMP position gain
    kp_zmp_vel_ = 10.0;      // ZMP velocity gain
    max_correction_ = 0.05;   // Maximum correction (5cm)
    ankle_gain_ = 0.1;        // Ankle control gain
    hip_gain_ = 0.05;         // Hip control gain
    arm_gain_ = 0.02;         // Arm control gain
    
    // Strategy selection
    use_ankle_strategy_ = true;
    use_hip_strategy_ = true;
    use_arm_strategy_ = true;
    
    // Foot geometry
    foot_length_ = 0.25;  // 25cm
    foot_width_ = 0.10;   // 10cm
    support_surface_z_ = 0.0;  // Ground level
  }

  // ZMP calculation parameters
  double gravity_;
  double com_height_;
  double support_surface_z_;
  std::vector<Vector2d> support_polygon_;
  Vector2d support_polygon_center_;
  
  // Balance control parameters
  double kp_zmp_pos_;
  double kp_zmp_vel_;
  double max_correction_;
  double ankle_gain_;
  double hip_gain_;
  double arm_gain_;
  
  // Foot geometry
  double foot_length_;
  double foot_width_;
  
  // Control strategies
  bool use_ankle_strategy_;
  bool use_hip_strategy_;
  bool use_arm_strategy_;
  bool standing_mode_ = true;
  
  // Commands
  double ankle_roll_command_ = 0.0;
  double ankle_pitch_command_ = 0.0;
  
  // Communication channels
  ISAAC_PROTO_RX(JointStatesProto, joint_states);
  ISAAC_PROTO_RX(ImuDataProto, imu_data);
  ISAAC_PROTO_TX(ZmpDataProto, zmp_data);
  ISAAC_PROTO_TX(JointCommandsProto, balance_joint_commands);
};

}  // namespace humanoid
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::humanoid::BalanceController);
```

## Walking Pattern Generation

### Preview Control for Walking
Generating stable walking patterns:

```cpp
#include "engine/alice/alice.hpp"
#include "engine/gems/walking/walking_pattern_generator.hpp"

namespace isaac {
namespace humanoid {

class WalkingPatternGenerator : public Codelet {
 public:
  void start() override {
    // Initialize walking parameters
    initializeWalkingParameters();
    
    // Set up preview controller
    setupPreviewController();
    
    tickOnMessage(rx_walking_command());
  }

  void tick() override {
    if (rx_walking_command().available()) {
      auto command = rx_walking_command().popLatest().value();
      
      // Update walking state based on command
      updateWalkingState(command);
      
      // Generate walking pattern
      auto pattern = generateWalkingPattern();
      
      // Publish footstep plan
      publishFootstepPlan(pattern);
      
      // Publish ZMP reference
      publishZMPReference(pattern);
    }
  }

 private:
  void initializeWalkingParameters() {
    // Walking parameters
    step_length_ = 0.3;        // 30cm step
    step_width_ = 0.2;         // 20cm step width
    step_height_ = 0.05;       // 5cm step height
    step_duration_ = 1.0;      // 1 second per step
    dsp_duration_ = 0.2;       // 0.2 seconds double support
   ssp_duration_ = step_duration_ - dsp_duration_; // Single support duration
    
    // Initialize state
    current_support_foot_ = SupportFoot::LEFT;
    walking_state_ = WalkingState::STANDING;
  }
  
  void setupPreviewController() {
    // Set up preview control for ZMP tracking
    // This involves solving the Riccati equation offline
    // and storing the solution for online use
    
    // Discretize the inverted pendulum model
    double dt = 0.01;  // 100 Hz control
    double omega = sqrt(gravity_ / com_height_);
    
    // State space matrices for inverted pendulum
    Matrix2d A;
    A << 1, dt, 
         omega*omega*dt, 1;
         
    Vector2d B;
    B << 0, -omega*omega*dt;
    
    // Cost matrices
    Matrix2d Q = Matrix2d::Identity() * 10.0;  // State penalty
    double R = 1.0;  // Control penalty
    
    // Solve Riccati equation to get feedback gains
    // This is a simplified version - in practice, use a proper solver
    Matrix2d P = solveRiccatiEquation(A, B, Q, R);
    
    // Compute feedback gain
    K_preview_ = (B.transpose() * P * B + R).inverse() * B.transpose() * P * A;
    
    // Set preview horizon
    preview_horizon_ = static_cast<int>(2.0 / dt);  // 2 second preview
  }
  
  WalkingPattern generateWalkingPattern() {
    WalkingPattern pattern;
    
    switch (walking_state_) {
      case WalkingState::STARTING:
        generateStartingPattern(pattern);
        break;
      case WalkingState::WALKING:
        generateWalkingPattern(pattern);
        break;
      case WalkingState::STOPPING:
        generateStoppingPattern(pattern);
        break;
      case WalkingState::STANDING:
        generateStandingPattern(pattern);
        break;
    }
    
    return pattern;
  }
  
  void generateWalkingPattern(WalkingPattern& pattern) {
    // Generate footsteps based on desired velocity
    Vector2d desired_velocity = getDesiredWalkingVelocity();
    
    // Compute step timing based on velocity
    double adjusted_step_duration = computeAdjustedStepDuration(desired_velocity.norm());
    
    // Generate footstep positions
    std::vector<Footstep> footsteps = computeFootsteps(desired_velocity, adjusted_step_duration);
    
    // Generate ZMP reference trajectory
    std::vector<Vector2d> zmp_refs = generateZMPReference(footsteps, adjusted_step_duration);
    
    // Generate CoM trajectory
    std::vector<Vector3d> com_refs = generateCoMReference(zmp_refs);
    
    // Package into pattern
    pattern.set_footsteps(footsteps);
    pattern.set_zmp_reference(zmp_refs);
    pattern.set_com_reference(com_refs);
    pattern.set_timing(adjusted_step_duration);
  }
  
  std::vector<Vector2d> generateZMPReference(const std::vector<Footstep>& footsteps,
                                           double step_duration) {
    std::vector<Vector2d> zmp_refs;
    
    // Generate ZMP trajectory that transitions between footholds
    for (size_t i = 0; i < footsteps.size(); i++) {
      Vector2d start_zmp, end_zmp;
      
      if (i == 0) {
        // Start at current ZMP position
        start_zmp = getCurrentZMP();
      } else {
        // Start at previous foothold center
        start_zmp = footsteps[i-1].position.head<2>();
      }
      
      // End at current foothold center
      end_zmp = footsteps[i].position.head<2>();
      
      // Generate smooth transition
      int steps_in_transition = static_cast<int>(step_duration / control_dt_);
      
      for (int j = 0; j < steps_in_transition; j++) {
        double t = static_cast<double>(j) / steps_in_transition;
        
        // Use cubic interpolation for smooth transition
        double interp_factor = 3*t*t - 2*t*t*t;  // Smooth interpolation
        Vector2d zmp_ref = start_zmp + interp_factor * (end_zmp - start_zmp);
        
        zmp_refs.push_back(zmp_ref);
      }
    }
    
    return zmp_refs;
  }
  
  void updateWalkingState(const WalkingCommand& command) {
    switch (command.mode()) {
      case WalkingCommand::START:
        if (walking_state_ == WalkingState::STANDING) {
          walking_state_ = WalkingState::STARTING;
        }
        break;
      case WalkingCommand::WALK:
        if (walking_state_ == WalkingState::STARTING || walking_state_ == WalkingState::WALKING) {
          walking_state_ = WalkingState::WALKING;
        }
        break;
      case WalkingCommand::STOP:
        if (walking_state_ == WalkingState::WALKING) {
          walking_state_ = WalkingState::STOPPING;
        }
        break;
      case WalkingCommand::STAND:
        walking_state_ = WalkingState::STANDING;
        break;
    }
  }

  enum class SupportFoot { LEFT, RIGHT };
  enum class WalkingState { STANDING, STARTING, WALKING, STOPPING };
  
  // Walking parameters
  double step_length_;
  double step_width_;
  double step_height_;
  double step_duration_;
  double dsp_duration_;
  double ssp_duration_;
  double com_height_ = 0.8;
  double gravity_ = 9.81;
  double control_dt_ = 0.01;
  
  // Preview control
  MatrixXd K_preview_;
  int preview_horizon_;
  
  // State
  SupportFoot current_support_foot_;
  WalkingState walking_state_;
  
  // Communication channels
  ISAAC_PROTO_RX(WalkingCommandProto, walking_command);
  ISAAC_PROTO_TX(FootstepPlanProto, footstep_plan);
  ISAAC_PROTO_TX(ZmpReferenceProto, zmp_reference);
};

}  // namespace humanoid
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::humanoid::WalkingPatternGenerator);
```

## Best Practices

### Kinematic Design Considerations
1. **Redundancy**: Design with extra DOFs for improved dexterity and obstacle avoidance
2. **Workspace**: Ensure adequate workspace for intended tasks
3. **Dexterity**: Optimize link lengths for desired manipulation capabilities
4. **Balancing**: Consider the effect of kinematic design on balance control

### Dynamic Modeling
1. **Accuracy**: Use accurate inertial parameters from CAD models or system identification
2. **Complexity**: Balance model complexity with computational requirements
3. **Validation**: Experimentally validate dynamic models
4. **Adaptation**: Implement adaptive control to compensate for model errors

### Balance Control
1. **Multi-strategy**: Combine ankle, hip, and stepping strategies
2. **Stability**: Ensure ZMP remains within support polygon
3. **Smoothness**: Use smooth transitions between different control strategies
4. **Reactivity**: Respond quickly to disturbances while maintaining stability

## Summary

Humanoid kinematics and dynamics form the foundation for controlling these complex robots. Understanding both forward and inverse kinematics is essential for task execution, while dynamic modeling enables stable locomotion and balance control. The Zero Moment Point concept is crucial for achieving stable walking, and preview control allows for predictive balance adjustments.

## Exercises

1. Implement forward kinematics for a simplified humanoid arm.
2. Derive the inverse kinematics solution for a 6-DOF humanoid leg.
3. Compute the mass matrix for a simple 2-link manipulator.
4. Implement a basic ZMP-based balance controller.