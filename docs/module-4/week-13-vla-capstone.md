---
sidebar_position: 3
---

# Week 13: Vision-Language-Action (VLA) & Conversational Robotics

## Learning Objectives

By the end of this week, you will be able to:
- Understand Vision-Language-Action (VLA) models and their applications in robotics
- Implement voice-to-action systems for humanoid robots
- Design cognitive planning architectures that translate natural language to robot actions
- Integrate speech recognition, language understanding, and action execution
- Create conversational interfaces for robot interaction
- Develop a complete capstone project: an autonomous humanoid with voice commands

## Introduction to Vision-Language-Action (VLA) Models

### Understanding VLA in Robotics
Vision-Language-Action (VLA) models represent a new paradigm in robotics that integrates perception, language understanding, and action execution in a unified framework. These models enable robots to understand natural language commands and execute appropriate actions based on visual perception of the environment.

### Key Components of VLA Systems
- **Vision Processing**: Understanding the visual environment
- **Language Understanding**: Interpreting natural language commands
- **Action Generation**: Mapping understood commands to executable robot actions
- **Multimodal Integration**: Combining vision and language information

### VLA Architecture
```cpp
#include "engine/alice/alice.hpp"
#include "engine/gems/vla/vla_model.hpp"

namespace isaac {
namespace vla {

class VLAModel : public Codelet {
 public:
  void start() override {
    // Initialize VLA model components
    initializeVisionEncoder();
    initializeLanguageEncoder();
    initializeActionDecoder();
    initializeMultimodalFusion();
    
    // Load pre-trained model weights
    loadModelWeights();
    
    // Set up inference pipeline
    setupInferencePipeline();
    
    tickOnMessage(rx_command());
  }

  void tick() override {
    if (rx_command().available()) {
      auto command_msg = rx_command().popLatest().value();
      
      // Get visual input
      auto visual_input = rx_visual_input().popLatest();
      if (!visual_input) return;
      
      // Process vision and language inputs
      auto vision_features = encodeVision(visual_input.value());
      auto language_features = encodeLanguage(command_msg.command());
      
      // Fuse multimodal features
      auto fused_features = fuseModalities(vision_features, language_features);
      
      // Generate action sequence
      auto action_sequence = decodeActions(fused_features);
      
      // Validate and refine actions
      auto validated_actions = validateActions(action_sequence, visual_input.value());
      
      // Publish action plan
      publishActionPlan(validated_actions);
    }
  }

 private:
  void initializeVisionEncoder() {
    // Initialize CNN-based vision encoder
    vision_encoder_ = std::make_unique<CNNEncoder>();
    vision_encoder_->loadModel("models/vision/resnet50.pt");
    vision_feature_dim_ = 2048;  // ResNet-50 feature dimension
  }
  
  void initializeLanguageEncoder() {
    // Initialize transformer-based language encoder
    language_encoder_ = std::make_unique<TransformerEncoder>();
    language_encoder_->loadModel("models/language/bert.pt");
    language_feature_dim_ = 768;  // BERT feature dimension
  }
  
  void initializeActionDecoder() {
    // Initialize action decoder
    action_decoder_ = std::make_unique<ActionDecoder>();
    action_decoder_->initialize(action_space_dim_);
  }
  
  void initializeMultimodalFusion() {
    // Initialize multimodal fusion layer
    fusion_layer_ = std::make_unique<MultimodalFusion>();
    fusion_layer_->initialize(vision_feature_dim_, language_feature_dim_);
  }
  
  VisionFeatures encodeVision(const VisualInput& input) {
    // Preprocess visual input
    auto preprocessed = preprocessVision(input);
    
    // Extract features using vision encoder
    return vision_encoder_->encode(preprocessed);
  }
  
  LanguageFeatures encodeLanguage(const std::string& command) {
    // Tokenize and encode language command
    auto tokens = tokenize(command);
    return language_encoder_->encode(tokens);
  }
  
  MultimodalFeatures fuseModalities(const VisionFeatures& vision, 
                                 const LanguageFeatures& language) {
    // Combine vision and language features
    return fusion_layer_->fuse(vision, language);
  }
  
  ActionSequence decodeActions(const MultimodalFeatures& features) {
    // Generate action sequence from fused features
    return action_decoder_->decode(features);
  }
  
  ActionSequence validateActions(const ActionSequence& raw_actions,
                              const VisualInput& visual_input) {
    // Validate actions against current visual state
    ActionSequence validated_actions;
    
    for (const auto& action : raw_actions) {
      if (validateAction(action, visual_input)) {
        validated_actions.push_back(action);
      } else {
        // Generate alternative action if validation fails
        auto alternative = generateAlternativeAction(action, visual_input);
        validated_actions.push_back(alternative);
      }
    }
    
    return validated_actions;
  }
  
  bool validateAction(const Action& action, const VisualInput& visual_input) {
    // Check if action is physically possible given current state
    switch (action.type) {
      case ActionType::NAVIGATE_TO:
        return isNavigable(action.target_position, visual_input);
      case ActionType::PICK_UP:
        return isGraspable(action.target_object, visual_input);
      case ActionType::PLACE:
        return isPlacable(action.target_location, visual_input);
      default:
        return true;  // Assume other actions are valid
    }
  }

  // Model components
  std::unique_ptr<CNNEncoder> vision_encoder_;
  std::unique_ptr<TransformerEncoder> language_encoder_;
  std::unique_ptr<ActionDecoder> action_decoder_;
  std::unique_ptr<MultimodalFusion> fusion_layer_;
  
  // Dimensions
  int vision_feature_dim_;
  int language_feature_dim_;
  int action_space_dim_ = 10;  // Example action space dimension
  
  // Communication channels
  ISAAC_PROTO_RX(CommandProto, command);
  ISAAC_PROTO_RX(VisualInputProto, visual_input);
  ISAAC_PROTO_TX(ActionPlanProto, action_plan);
};

}  // namespace vla
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::vla::VLAModel);
```

## Voice-to-Action Systems

### Speech Recognition Integration
Implementing speech-to-text conversion for voice commands:

```cpp
#include "engine/alice/alice.hpp"
#include "engine/gems/speech/speech_recognizer.hpp"

namespace isaac {
namespace speech {

class VoiceToAction : public Codelet {
 public:
  void start() override {
    // Initialize speech recognizer
    initializeSpeechRecognizer();
    
    // Set up wake word detection
    setupWakeWordDetection();
    
    // Initialize language understanding module
    initializeLanguageUnderstanding();
    
    // Start listening for voice commands
    startListening();
  }

  void tick() override {
    // Process audio input
    auto audio_input = rx_audio_input().popLatest();
    if (audio_input) {
      // Check for wake word
      if (detectWakeWord(audio_input.value())) {
        // Start recording command after wake word
        startCommandRecording();
      }
      
      // Process ongoing recording
      if (isRecording_) {
        accumulateAudioBuffer(audio_input.value());
        
        // Check if command is complete (silence detection)
        if (isCommandComplete(audio_input.value())) {
          // Convert speech to text
          std::string command_text = recognizeSpeech(recorded_audio_);
          
          // Process the command
          processCommand(command_text);
          
          // Reset for next command
          stopCommandRecording();
        }
      }
    }
  }

 private:
  void initializeSpeechRecognizer() {
    // Initialize speech recognition model
    speech_recognizer_ = std::make_unique<SpeechRecognizer>();
    
#ifdef USE_OFFLINE_RECOGNITION
    // Use offline model (e.g., Whisper)
    speech_recognizer_->loadModel("models/speech/whisper_small.pt");
#else
    // Use online API (e.g., Google Speech-to-Text)
    speech_recognizer_->initializeAPI("GOOGLE_STT_API_KEY");
#endif
    
    // Set up audio preprocessing
    audio_preprocessor_.setSampleRate(16000);
    audio_preprocessor_.setChannels(1);
  }
  
  void setupWakeWordDetection() {
    // Initialize wake word detection
    wake_word_detector_ = std::make_unique<WakeWordDetector>();
    wake_word_detector_->setWakeWord("ROBOT");
    wake_word_detector_->setThreshold(0.8);
  }
  
  void initializeLanguageUnderstanding() {
    // Initialize NLU (Natural Language Understanding) module
    nlu_module_ = std::make_unique<NaturalLanguageUnderstanding>();
    nlu_module_->loadIntentModel("models/nlu/intent_classifier.pt");
    nlu_module_->loadEntityModel("models/nlu/entity_extractor.pt");
  }
  
  bool detectWakeWord(const AudioBuffer& audio) {
    return wake_word_detector_->detect(audio);
  }
  
  void startCommandRecording() {
    isRecording_ = true;
    recorded_audio_.clear();
    silence_start_time_ = 0;
    command_timeout_start_ = getNodeTime();
  }
  
  void accumulateAudioBuffer(const AudioBuffer& audio) {
    recorded_audio_.insert(recorded_audio_.end(), 
                          audio.samples.begin(), 
                          audio.samples.end());
  }
  
  bool isCommandComplete(const AudioBuffer& audio) {
    // Check for silence (indicating end of command)
    double volume = calculateVolume(audio);
    
    if (volume < silence_threshold_) {
      if (silence_start_time_ == 0) {
        silence_start_time_ = getNodeTime();
      } else if (getNodeTime() - silence_start_time_ > silence_duration_threshold_) {
        return true;
      }
    } else {
      silence_start_time_ = 0;
    }
    
    // Check for timeout
    if (getNodeTime() - command_timeout_start_ > command_timeout_threshold_) {
      return true;
    }
    
    return false;
  }
  
  std::string recognizeSpeech(const std::vector<float>& audio_buffer) {
    // Convert audio to text
    return speech_recognizer_->recognize(audio_buffer);
  }
  
  void processCommand(const std::string& command_text) {
    // Parse the command using NLU
    auto parsed_command = nlu_module_->parse(command_text);
    
    // Validate the command
    if (validateCommand(parsed_command)) {
      // Publish the command for action planning
      publishParsedCommand(parsed_command);
      
      // Provide audio feedback
      provideAudioFeedback("Processing command: " + command_text);
    } else {
      // Invalid command
      provideAudioFeedback("Invalid command: " + command_text);
    }
  }
  
  bool validateCommand(const ParsedCommand& command) {
    // Validate command structure and semantics
    if (command.intent.empty()) {
      return false;
    }
    
    // Check if required entities are present
    if (command.intent == "NAVIGATE" && command.entities.count("location") == 0) {
      return false;
    }
    
    if (command.intent == "PICK_UP" && command.entities.count("object") == 0) {
      return false;
    }
    
    return true;
  }
  
  void provideAudioFeedback(const std::string& feedback) {
    // Generate audio feedback (text-to-speech)
    auto tts_output = textToSpeech(feedback);
    tx_audio_feedback().initProto().set_samples(tts_output);
    tx_audio_feedback().publish();
  }

  // Speech recognition components
  std::unique_ptr<SpeechRecognizer> speech_recognizer_;
  std::unique_ptr<WakeWordDetector> wake_word_detector_;
  std::unique_ptr<NaturalLanguageUnderstanding> nlu_module_;
  AudioPreprocessor audio_preprocessor_;
  
  // State variables
  bool isRecording_ = false;
  std::vector<float> recorded_audio_;
  double silence_start_time_ = 0;
  double command_timeout_start_ = 0;
  
  // Thresholds
  double silence_threshold_ = 0.01;
  double silence_duration_threshold_ = 1.0;  // 1 second of silence
  double command_timeout_threshold_ = 10.0;  // 10 second timeout
  
  // Communication channels
  ISAAC_PROTO_RX(AudioBufferProto, audio_input);
  ISAAC_PROTO_TX(ParsedCommandProto, parsed_command);
  ISAAC_PROTO_TX(AudioBufferProto, audio_feedback);
};

}  // namespace speech
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::speech::VoiceToAction);
```

## Cognitive Planning Architecture

### Language-to-Action Translation
Creating a cognitive planner that translates natural language to robot actions:

```cpp
#include "engine/alice/alice.hpp"
#include "engine/gems/planning/cognitive_planner.hpp"

namespace isaac {
namespace planning {

class CognitivePlanner : public Codelet {
 public:
  void start() override {
    // Initialize action library
    initializeActionLibrary();
    
    // Set up task decomposition
    setupTaskDecomposer();
    
    // Initialize world model
    initializeWorldModel();
    
    // Set up plan validator
    setupPlanValidator();
    
    tickOnMessage(rx_parsed_command());
  }

  void tick() override {
    if (rx_parsed_command().available()) {
      auto parsed_cmd = rx_parsed_command().popLatest().value();
      
      // Decompose high-level command into subtasks
      auto task_sequence = decomposeTask(parsed_cmd);
      
      // Generate detailed action plan
      auto action_plan = generateActionPlan(task_sequence);
      
      // Validate the plan
      if (validatePlan(action_plan)) {
        // Publish the plan
        publishActionPlan(action_plan);
        
        // Update world model expectations
        updateWorldModelPredictions(action_plan);
      } else {
        // Plan validation failed, generate error response
        generateErrorResponse("Could not generate valid plan for command");
      }
    }
  }

 private:
  void initializeActionLibrary() {
    // Define primitive actions available to the robot
    action_library_["NAVIGATE_TO"] = {
      .preconditions = {"robot_is_operational"},
      .effects = {"robot_at_location"},
      .implementation = "navigation_stack"
    };
    
    action_library_["PICK_UP"] = {
      .preconditions = {"object_is_reachable", "gripper_is_open"},
      .effects = {"object_is_grasped", "gripper_is_closed"},
      .implementation = "manipulation_stack"
    };
    
    action_library_["PLACE"] = {
      .preconditions = {"object_is_grasped", "location_is_free"},
      .effects = {"object_is_placed", "gripper_is_open"},
      .implementation = "manipulation_stack"
    };
    
    action_library_["FIND_OBJECT"] = {
      .preconditions = {"robot_can_move", "object_exists"},
      .effects = {"object_location_known"},
      .implementation = "perception_stack"
    };
    
    action_library_["ANSWER_QUESTION"] = {
      .preconditions = {"question_understood"},
      .effects = {"information_provided"},
      .implementation = "dialogue_manager"
    };
  }
  
  TaskSequence decomposeTask(const ParsedCommand& command) {
    TaskSequence tasks;
    
    // Map intents to task sequences
    if (command.intent == "CLEAN_ROOM") {
      // Complex task: "Clean the room"
      tasks = {
        {"FIND_DUSTY_SPOT", {}},
        {"NAVIGATE_TO", {{"location", "dusty_spot"}}},
        {"CLEAN_AREA", {}},
        {"FIND_NEXT_SPOT", {}}
      };
    } else if (command.intent == "BRING_ME_COFFEE") {
      // Complex task: "Bring me coffee"
      tasks = {
        {"FIND_OBJECT", {{"object", "coffee"}}},
        {"NAVIGATE_TO", {{"location", "coffee_location"}}},
        {"PICK_UP", {{"object", "coffee"}}},
        {"NAVIGATE_TO", {{"location", "user_location"}}},
        {"PLACE", {{"location", "delivery_location"}}}
      };
    } else if (command.intent == "ANSWER_QUESTION") {
      // Question answering task
      tasks = {
        {"UNDERSTAND_QUESTION", {{"question", command.raw_text}}},
        {"SEARCH_KNOWLEDGE", {}},
        {"FORMULATE_ANSWER", {}},
        {"ANSWER_QUESTION", {}}
      };
    } else {
      // Simple direct action
      tasks = {{command.intent, command.entities}};
    }
    
    return tasks;
  }
  
  ActionPlan generateActionPlan(const TaskSequence& tasks) {
    ActionPlan plan;
    
    for (const auto& task : tasks) {
      // Get action definition
      auto action_def = action_library_[task.type];
      
      // Generate specific action instance
      ActionInstance action;
      action.type = task.type;
      action.parameters = task.parameters;
      action.preconditions = action_def.preconditions;
      action.effects = action_def.effects;
      
      // Add to plan
      plan.actions.push_back(action);
    }
    
    return plan;
  }
  
  bool validatePlan(const ActionPlan& plan) {
    // Check if plan is logically consistent
    auto initial_state = getCurrentWorldState();
    
    for (const auto& action : plan.actions) {
      // Check preconditions
      if (!satisfiesPreconditions(initial_state, action.preconditions)) {
        return false;
      }
      
      // Apply effects to state
      initial_state = applyEffects(initial_state, action.effects);
    }
    
    return true;
  }
  
  WorldState getCurrentWorldState() {
    // Get current world state from world model
    WorldState state;
    
    // Populate with known facts
    state.facts.insert("robot_is_operational");
    state.facts.insert("gripper_is_open");
    
    // Add location information
    auto robot_pose = getRobotPose();
    state.properties["robot_location"] = robot_pose;
    
    // Add object locations
    auto objects = getKnownObjects();
    for (const auto& obj : objects) {
      state.properties[obj.name + "_location"] = obj.pose;
    }
    
    return state;
  }
  
  bool satisfiesPreconditions(const WorldState& state, 
                           const std::vector<std::string>& preconditions) {
    for (const auto& precondition : preconditions) {
      if (state.facts.find(precondition) == state.facts.end()) {
        return false;
      }
    }
    return true;
  }
  
  WorldState applyEffects(const WorldState& state, 
                        const std::vector<std::string>& effects) {
    WorldState new_state = state;
    
    for (const auto& effect : effects) {
      // Add positive effects
      new_state.facts.insert(effect);
      
      // Remove contradictory facts (simple STRIPS-style)
      std::string negated_effect = "NOT_" + effect;
      new_state.facts.erase(negated_effect);
    }
    
    return new_state;
  }
  
  void updateWorldModelPredictions(const ActionPlan& plan) {
    // Update world model with expected changes
    WorldState predicted_state = getCurrentWorldState();
    
    for (const auto& action : plan.actions) {
      predicted_state = applyEffects(predicted_state, action.effects);
    }
    
    // Store predictions for later validation
    expected_world_state_ = predicted_state;
  }

  // Action library
  std::map<std::string, ActionDefinition> action_library_;
  
  // World model
  WorldState expected_world_state_;
  
  // Communication channels
  ISAAC_PROTO_RX(ParsedCommandProto, parsed_command);
  ISAAC_PROTO_TX(ActionPlanProto, action_plan);
};

}  // namespace planning
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::planning::CognitivePlanner);
```

## Conversational Interface

### Dialogue Management System
Creating a conversational interface for natural interaction:

```cpp
#include "engine/alice/alice.hpp"
#include "engine/gems/dialogue/dialogue_manager.hpp"

namespace isaac {
namespace dialogue {

class ConversationalInterface : public Codelet {
 public:
  void start() override {
    // Initialize dialogue manager
    initializeDialogueManager();
    
    // Set up conversation history
    setupConversationHistory();
    
    // Initialize context tracking
    initializeContextTracker();
    
    tickOnMessage(rx_user_input());
  }

  void tick() override {
    if (rx_user_input().available()) {
      auto user_input = rx_user_input().popLatest().value();
      
      // Update conversation context
      updateContext(user_input);
      
      // Generate appropriate response
      auto response = generateResponse(user_input);
      
      // Send response
      sendResponse(response);
      
      // Update conversation history
      updateConversationHistory(user_input, response);
    }
  }

 private:
  void initializeDialogueManager() {
    // Initialize dialogue state machine
    dialogue_state_ = DialogueState::IDLE;
    
    // Load dialogue policy
    dialogue_policy_ = std::make_unique<DialoguePolicy>();
    dialogue_policy_->loadModel("models/dialogue/dialogue_policy.pt");
    
    // Initialize response generator
    response_generator_ = std::make_unique<ResponseGenerator>();
    response_generator_->loadModel("models/dialogue/response_gen.pt");
  }
  
  void setupConversationHistory() {
    // Initialize conversation history buffer
    conversation_history_.max_size = 10;  // Keep last 10 exchanges
    conversation_history_.exchanges.clear();
  }
  
  void initializeContextTracker() {
    // Initialize context variables
    context_variables_["current_task"] = "";
    context_variables_["user_intent"] = "";
    context_variables_["object_focus"] = "";
    context_variables_["location_focus"] = "";
  }
  
  void updateContext(const UserInput& input) {
    // Update context based on user input
    if (input.type == InputType::COMMAND) {
      context_variables_["current_task"] = input.parsed_command.intent;
      dialogue_state_ = DialogueState::TASK_EXECUTION;
    } else if (input.type == InputType::QUESTION) {
      dialogue_state_ = DialogueState::ANSWERING_QUESTION;
    } else if (input.type == InputType::CHAT) {
      dialogue_state_ = DialogueState::SOCIAL_INTERACTION;
    }
    
    // Update entity focus
    if (!input.parsed_command.entities.empty()) {
      for (const auto& entity : input.parsed_command.entities) {
        if (entity.first == "object") {
          context_variables_["object_focus"] = entity.second;
        } else if (entity.first == "location") {
          context_variables_["location_focus"] = entity.second;
        }
      }
    }
  }
  
  Response generateResponse(const UserInput& input) {
    Response response;
    
    switch (dialogue_state_) {
      case DialogueState::IDLE:
        response.type = ResponseType::GREETING;
        response.text = getGreetingResponse();
        break;
        
      case DialogueState::TASK_EXECUTION:
        response.type = ResponseType::TASK_CONFIRMATION;
        response.text = getTaskConfirmation(input);
        break;
        
      case DialogueState::ANSWERING_QUESTION:
        response.type = ResponseType::INFORMATION;
        response.text = getInformationResponse(input);
        break;
        
      case DialogueState::SOCIAL_INTERACTION:
        response.type = ResponseType::CONVERSATIONAL;
        response.text = getConversationalResponse(input);
        break;
        
      case DialogueState::ERROR:
        response.type = ResponseType::ERROR_MESSAGE;
        response.text = getErrorResponse(input);
        break;
    }
    
    return response;
  }
  
  std::string getGreetingResponse() {
    // Generate greeting based on time/context
    auto current_time = std::time(nullptr);
    auto time_info = std::localtime(&current_time);
    
    if (time_info->tm_hour < 12) {
      return "Good morning! How can I assist you today?";
    } else if (time_info->tm_hour < 18) {
      return "Good afternoon! How can I help you?";
    } else {
      return "Good evening! What would you like me to do?";
    }
  }
  
  std::string getTaskConfirmation(const UserInput& input) {
    // Confirm the task to be executed
    std::string task = input.parsed_command.intent;
    
    if (task == "NAVIGATE_TO") {
      auto location = input.parsed_command.entities.at("location");
      return "I will navigate to the " + location + ". Please wait.";
    } else if (task == "PICK_UP") {
      auto object = input.parsed_command.entities.at("object");
      return "I will pick up the " + object + ". Please wait.";
    } else if (task == "BRING_ME") {
      auto object = input.parsed_command.entities.at("object");
      return "I will bring you the " + object + ". Please wait.";
    }
    
    return "I will execute the requested task. Please wait.";
  }
  
  std::string getInformationResponse(const UserInput& input) {
    // Answer questions based on world knowledge
    std::string question = input.raw_text;
    
    // Simple keyword-based response for demonstration
    if (question.find("weather") != std::string::npos) {
      return "I don't have access to weather information, but I can help with other tasks.";
    } else if (question.find("time") != std::string::npos) {
      auto current_time = std::time(nullptr);
      char time_str[100];
      std::strftime(time_str, sizeof(time_str), "%H:%M", std::localtime(&current_time));
      return "The current time is " + std::string(time_str) + ".";
    } else if (question.find("your name") != std::string::npos) {
      return "I am your AI-powered humanoid assistant. You can call me ARIA (Autonomous Robot Intelligence Assistant).";
    } else {
      return "I'm not sure about that. I can help with navigation, manipulation, and other tasks. How else can I assist you?";
    }
  }
  
  std::string getConversationalResponse(const UserInput& input) {
    // Generate social responses
    std::string text = input.raw_text;
    
    if (text.find("hello") != std::string::npos || 
        text.find("hi") != std::string::npos) {
      return "Hello! It's great to see you. How can I help?";
    } else if (text.find("thank") != std::string::npos) {
      return "You're welcome! I'm happy to help.";
    } else if (text.find("how are you") != std::string::npos) {
      return "I'm functioning optimally, thank you for asking! How can I assist you today?";
    } else {
      return "That's interesting. I'm here to help with tasks like cleaning, fetching items, or navigating. Is there something specific you need?";
    }
  }
  
  void sendResponse(const Response& response) {
    // Send response via text-to-speech
    auto tts_output = textToSpeech(response.text);
    
    // Publish audio response
    auto audio_msg = tx_audio_response().initProto();
    audio_msg.set_samples(tts_output);
    audio_msg.set_text(response.text);
    tx_audio_response().publish();
    
    // Publish text response for display
    auto text_msg = tx_text_response().initProto();
    text_msg.set_text(response.text);
    tx_text_response().publish();
  }
  
  void updateConversationHistory(const UserInput& user_input, 
                               const Response& response) {
    // Add exchange to history
    ConversationExchange exchange;
    exchange.user_input = user_input.raw_text;
    exchange.response = response.text;
    exchange.timestamp = getNodeTime();
    
    conversation_history_.exchanges.push_back(exchange);
    
    // Trim history if too long
    if (conversation_history_.exchanges.size() > conversation_history_.max_size) {
      conversation_history_.exchanges.erase(conversation_history_.exchanges.begin());
    }
  }

  enum class DialogueState {
    IDLE,
    TASK_EXECUTION,
    ANSWERING_QUESTION,
    SOCIAL_INTERACTION,
    ERROR
  };
  
  // Dialogue components
  std::unique_ptr<DialoguePolicy> dialogue_policy_;
  std::unique_ptr<ResponseGenerator> response_generator_;
  
  // State
  DialogueState dialogue_state_;
  std::map<std::string, std::string> context_variables_;
  
  // Conversation history
  struct ConversationHistory {
    std::vector<ConversationExchange> exchanges;
    int max_size;
  } conversation_history_;
  
  // Communication channels
  ISAAC_PROTO_RX(UserInputProto, user_input);
  ISAAC_PROTO_TX(AudioResponseProto, audio_response);
  ISAAC_PROTO_TX(TextResponseProto, text_response);
};

}  // namespace dialogue
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::dialogue::ConversationalInterface);
```

## Capstone Project: Autonomous Humanoid with Voice Commands

### Complete System Integration
Bringing together all components for the capstone project:

```cpp
#include "engine/alice/alice.hpp"
#include "engine/gems/integration/humanoid_system.hpp"

namespace isaac {
namespace capstone {

class AutonomousHumanoid : public Codelet {
 public:
  void start() override {
    // Initialize all subsystems
    initializePerceptionSystem();
    initializeSpeechSystem();
    initializePlanningSystem();
    initializeControlSystem();
    initializeDialogueSystem();
    
    // Set up system monitoring
    setupSystemHealthMonitor();
    
    // Initialize with idle state
    system_state_ = SystemState::IDLE;
    
    tickPeriodically(std::chrono::milliseconds(10));  // 100 Hz main loop
  }

  void tick() override {
    // Monitor system health
    checkSystemHealth();
    
    // Handle different operational states
    switch (system_state_) {
      case SystemState::IDLE:
        handleIdleState();
        break;
      case SystemState::LISTENING:
        handleListeningState();
        break;
      case SystemState::PROCESSING_COMMAND:
        handleCommandProcessing();
        break;
      case SystemState::EXECUTING_TASK:
        handleTaskExecution();
        break;
      case SystemState::ERROR:
        handleErrorState();
        break;
    }
    
    // Update system status
    publishSystemStatus();
  }

 private:
  void initializePerceptionSystem() {
    // Initialize perception stack
    perception_system_ = std::make_unique<PerceptionSystem>();
    perception_system_->initialize();
    
    // Set up object detection
    object_detector_ = std::make_unique<ObjectDetector>();
    object_detector_->loadModel("models/perception/object_detector.pt");
    
    // Set up SLAM for navigation
    slam_system_ = std::make_unique<SLAMSystem>();
    slam_system_->initialize();
  }
  
  void initializeSpeechSystem() {
    // Initialize speech recognition and synthesis
    speech_system_ = std::make_unique<SpeechSystem>();
    speech_system_->initialize();
    
    // Set up wake word detection
    wake_word_detector_ = std::make_unique<WakeWordDetector>();
    wake_word_detector_->setWakeWord("ARIA");
  }
  
  void initializePlanningSystem() {
    // Initialize task and motion planners
    task_planner_ = std::make_unique<TaskPlanner>();
    task_planner_->initialize();
    
    motion_planner_ = std::make_unique<MotionPlanner>();
    motion_planner_->initialize();
  }
  
  void initializeControlSystem() {
    // Initialize robot controllers
    navigation_controller_ = std::make_unique<NavigationController>();
    navigation_controller_->initialize();
    
    manipulation_controller_ = std::make_unique<ManipulationController>();
    manipulation_controller_->initialize();
    
    balance_controller_ = std::make_unique<BalanceController>();
    balance_controller_->initialize();
  }
  
  void initializeDialogueSystem() {
    // Initialize conversational interface
    dialogue_system_ = std::make_unique<DialogueSystem>();
    dialogue_system_->initialize();
  }
  
  void handleIdleState() {
    // In idle state, listen for wake word
    auto audio_input = rx_audio_input().popLatest();
    if (audio_input && wake_word_detector_->detect(audio_input.value())) {
      // Wake word detected, transition to listening state
      system_state_ = SystemState::LISTENING;
      startCommandRecording();
    }
    
    // Also check for direct command input (e.g., from GUI)
    if (rx_direct_command().available()) {
      auto cmd = rx_direct_command().popLatest().value();
      processDirectCommand(cmd);
    }
  }
  
  void handleListeningState() {
    // Accumulate audio for command
    auto audio_input = rx_audio_input().popLatest();
    if (audio_input) {
      accumulateAudioBuffer(audio_input.value());
      
      // Check if command is complete
      if (isCommandComplete(audio_input.value())) {
        // Recognize speech
        std::string command_text = speech_system_->recognizeSpeech(recorded_audio_);
        
        // Process the command
        processSpokenCommand(command_text);
        
        // Return to idle state
        system_state_ = SystemState::PROCESSING_COMMAND;
      }
    }
    
    // Timeout check
    if (getNodeTime() - command_start_time_ > command_timeout_) {
      // Command timeout, return to idle
      system_state_ = SystemState::IDLE;
      stopCommandRecording();
    }
  }
  
  void handleCommandProcessing() {
    // Process the recognized command through the pipeline
    if (current_command_.empty()) {
      system_state_ = SystemState::IDLE;
      return;
    }
    
    // Parse command
    auto parsed_command = parseCommand(current_command_);
    
    if (!parsed_command.valid) {
      // Invalid command, provide feedback
      provideFeedback("I didn't understand that command. Please try again.");
      system_state_ = SystemState::IDLE;
      return;
    }
    
    // Plan the task
    auto task_plan = task_planner_->plan(parsed_command);
    
    if (!task_plan.valid) {
      // Planning failed, provide feedback
      provideFeedback("I couldn't plan how to do that. Can you rephrase?");
      system_state_ = SystemState::IDLE;
      return;
    }
    
    // Convert to motion plan
    auto motion_plan = motion_planner_->plan(task_plan);
    
    if (!motion_plan.valid) {
      // Motion planning failed
      provideFeedback("I can't physically do that task. Is there something else?");
      system_state_ = SystemState::IDLE;
      return;
    }
    
    // Execute the plan
    execution_status_ = ExecutionStatus::PENDING;
    executeMotionPlan(motion_plan);
    
    // Transition to execution state
    system_state_ = SystemState::EXECUTING_TASK;
  }
  
  void handleTaskExecution() {
    // Monitor execution progress
    auto status = getExecutionStatus();
    
    switch (status) {
      case ExecutionStatus::RUNNING:
        // Continue monitoring
        break;
      case ExecutionStatus::SUCCESS:
        // Task completed successfully
        provideFeedback("Task completed successfully!");
        system_state_ = SystemState::IDLE;
        break;
      case ExecutionStatus::FAILED:
        // Task failed
        provideFeedback("I couldn't complete that task. Would you like me to try again?");
        system_state_ = SystemState::IDLE;
        break;
      case ExecutionStatus::INTERRUPTED:
        // Task interrupted (e.g., by new command)
        provideFeedback("Task interrupted.");
        system_state_ = SystemState::IDLE;
        break;
    }
  }
  
  void handleErrorState() {
    // Handle system errors
    if (isSystemHealthy()) {
      // System recovered, return to idle
      system_state_ = SystemState::IDLE;
      provideFeedback("System recovered. Ready for commands.");
    } else {
      // Still in error state, continue monitoring
      if (getNodeTime() - error_start_time_ > error_recovery_timeout_) {
        // Attempt system restart
        attemptSystemRestart();
      }
    }
  }
  
  void checkSystemHealth() {
    // Check all subsystems
    bool perception_ok = perception_system_->isHealthy();
    bool speech_ok = speech_system_->isHealthy();
    bool planning_ok = task_planner_->isHealthy() && motion_planner_->isHealthy();
    bool control_ok = navigation_controller_->isHealthy() && 
                     manipulation_controller_->isHealthy() &&
                     balance_controller_->isHealthy();
    
    system_healthy_ = perception_ok && speech_ok && planning_ok && control_ok;
    
    if (!system_healthy_ && system_state_ != SystemState::ERROR) {
      // System became unhealthy, transition to error state
      system_state_ = SystemState::ERROR;
      error_start_time_ = getNodeTime();
      provideFeedback("System error detected. Attempting recovery...");
    }
  }
  
  void processSpokenCommand(const std::string& command) {
    current_command_ = command;
    command_start_time_ = getNodeTime();
  }
  
  void processDirectCommand(const DirectCommand& cmd) {
    // Process command that came through direct interface (not speech)
    current_command_ = cmd.text;
    system_state_ = SystemState::PROCESSING_COMMAND;
  }
  
  void provideFeedback(const std::string& message) {
    // Provide audio feedback
    auto tts_output = textToSpeech(message);
    
    auto audio_msg = tx_feedback_audio().initProto();
    audio_msg.set_samples(tts_output);
    audio_msg.set_text(message);
    tx_feedback_audio().publish();
    
    // Also send to text display
    auto text_msg = tx_feedback_text().initProto();
    text_msg.set_text(message);
    tx_feedback_text().publish();
  }
  
  void executeMotionPlan(const MotionPlan& plan) {
    // Send plan to controllers
    navigation_controller_->execute(plan.navigation_part);
    manipulation_controller_->execute(plan.manipulation_part);
    balance_controller_->execute(plan.balance_part);
  }

  enum class SystemState {
    IDLE,
    LISTENING,
    PROCESSING_COMMAND,
    EXECUTING_TASK,
    ERROR
  };
  
  enum class ExecutionStatus {
    PENDING,
    RUNNING,
    SUCCESS,
    FAILED,
    INTERRUPTED
  };
  
  // System components
  std::unique_ptr<PerceptionSystem> perception_system_;
  std::unique_ptr<ObjectDetector> object_detector_;
  std::unique_ptr<SLAMSystem> slam_system_;
  std::unique_ptr<SpeechSystem> speech_system_;
  std::unique_ptr<WakeWordDetector> wake_word_detector_;
  std::unique_ptr<TaskPlanner> task_planner_;
  std::unique_ptr<MotionPlanner> motion_planner_;
  std::unique_ptr<NavigationController> navigation_controller_;
  std::unique_ptr<ManipulationController> manipulation_controller_;
  std::unique_ptr<BalanceController> balance_controller_;
  std::unique_ptr<DialogueSystem> dialogue_system_;
  
  // State management
  SystemState system_state_;
  ExecutionStatus execution_status_;
  bool system_healthy_ = true;
  double error_start_time_ = 0.0;
  
  // Command processing
  std::string current_command_;
  std::vector<float> recorded_audio_;
  double command_start_time_ = 0.0;
  double command_timeout_ = 10.0;  // 10 seconds
  
  // Error handling
  double error_recovery_timeout_ = 30.0;  // 30 seconds
  
  // Communication channels
  ISAAC_PROTO_RX(AudioBufferProto, audio_input);
  ISAAC_PROTO_RX(DirectCommandProto, direct_command);
  ISAAC_PROTO_TX(SystemStatusProto, system_status);
  ISAAC_PROTO_TX(AudioBufferProto, feedback_audio);
  ISAAC_PROTO_TX(TextProto, feedback_text);
};

}  // namespace capstone
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::capstone::AutonomousHumanoid);
```

## System Integration and Testing

### Integration Testing Framework
Testing the complete VLA system:

```cpp
#include "engine/alice/alice.hpp"
#include "engine/gems/testing/integration_tester.hpp"

namespace isaac {
namespace testing {

class VLASystemTester : public Codelet {
 public:
  void start() override {
    // Initialize test scenarios
    initializeTestScenarios();
    
    // Set up test environment
    setupTestEnvironment();
    
    // Start automated testing
    startAutomatedTests();
  }

  void tick() override {
    // Run next test in sequence
    if (!test_queue_.empty() && !test_running_) {
      runNextTest();
    }
    
    // Monitor test progress
    if (test_running_) {
      monitorTestProgress();
    }
  }

 private:
  void initializeTestScenarios() {
    // Define test scenarios for VLA system
    test_scenarios_ = {
      {
        .name = "Simple Navigation",
        .input = "Go to the kitchen",
        .expected_actions = {"NAVIGATE_TO:kitchen"},
        .environment_setup = [](TestEnvironment& env) {
          env.addObject("robot", {0, 0, 0});
          env.addObject("kitchen", {5, 0, 0});
        }
      },
      {
        .name = "Object Manipulation",
        .input = "Pick up the red cup",
        .expected_actions = {"FIND_OBJECT:red_cup", "NAVIGATE_TO:red_cup", "PICK_UP:red_cup"},
        .environment_setup = [](TestEnvironment& env) {
          env.addObject("robot", {0, 0, 0});
          env.addObject("red_cup", {2, 1, 0.8});
        }
      },
      {
        .name = "Complex Task",
        .input = "Clean the table and bring me water",
        .expected_actions = {
          "NAVIGATE_TO:table", 
          "CLEAN_AREA:table", 
          "NAVIGATE_TO:kitchen", 
          "PICK_UP:water", 
          "NAVIGATE_TO:user", 
          "PLACE:water"
        },
        .environment_setup = [](TestEnvironment& env) {
          env.addObject("robot", {0, 0, 0});
          env.addObject("table", {3, 2, 0.8});
          env.addObject("kitchen", {5, 0, 0});
          env.addObject("water", {5.5, 0.5, 0.9});
          env.addObject("user", {1, 1, 0});
        }
      },
      {
        .name = "Question Answering",
        .input = "What time is it?",
        .expected_actions = {"ANSWER_QUESTION:time"},
        .environment_setup = [](TestEnvironment& env) {
          env.addObject("robot", {0, 0, 0});
        }
      }
    };
    
    // Queue all tests
    for (size_t i = 0; i < test_scenarios_.size(); i++) {
      test_queue_.push(i);
    }
  }
  
  void setupTestEnvironment() {
    // Initialize test environment
    test_env_.initialize();
    
    // Set up mock sensors and actuators
    setupMockComponents();
  }
  
  void runNextTest() {
    if (test_queue_.empty()) {
      // All tests completed
      finalizeTesting();
      return;
    }
    
    // Get next test
    size_t test_idx = test_queue_.front();
    test_queue_.pop();
    
    current_test_ = test_scenarios_[test_idx];
    
    // Set up environment for this test
    current_test_.environment_setup(test_env_);
    
    // Send command to system
    sendCommandToSystem(current_test_.input);
    
    // Start monitoring
    test_start_time_ = getNodeTime();
    test_running_ = true;
    actions_recorded_.clear();
    
    LOG_INFO("Running test: %s", current_test_.name.c_str());
  }
  
  void monitorTestProgress() {
    // Check if test has produced actions
    while (rx_system_actions().available()) {
      auto action = rx_system_actions().popLatest().value();
      actions_recorded_.push_back(action);
    }
    
    // Check for test completion
    bool test_completed = checkTestCompletion();
    
    if (test_completed || (getNodeTime() - test_start_time_) > test_timeout_) {
      // Test finished (either completed or timed out)
      evaluateTestResult();
      test_running_ = false;
    }
  }
  
  bool checkTestCompletion() {
    // For navigation tests, check if robot reached destination
    if (containsAction(actions_recorded_, "NAVIGATE_TO")) {
      // Check if navigation completed
      return isNavigationCompleted();
    }
    
    // For manipulation tests, check if object picked/placed
    if (containsAction(actions_recorded_, "PICK_UP") || 
        containsAction(actions_recorded_, "PLACE")) {
      return isManipulationCompleted();
    }
    
    // For question answering, check if response generated
    if (containsAction(actions_recorded_, "ANSWER_QUESTION")) {
      return isQuestionAnswered();
    }
    
    return false;
  }
  
  void evaluateTestResult() {
    // Compare recorded actions with expected actions
    bool passed = true;
    std::string result_details = "";
    
    // Check if all expected actions were performed
    for (const auto& expected : current_test_.expected_actions) {
      bool found = false;
      for (const auto& actual : actions_recorded_) {
        if (actual.contains(expected)) {
          found = true;
          break;
        }
      }
      
      if (!found) {
        passed = false;
        result_details += "Missing expected action: " + expected + "; ";
      }
    }
    
    // Check if any unexpected actions were performed
    for (const auto& actual : actions_recorded_) {
      bool expected = false;
      for (const auto& expected_act : current_test_.expected_actions) {
        if (actual.contains(expected_act)) {
          expected = true;
          break;
        }
      }
      
      if (!expected) {
        result_details += "Unexpected action: " + actual + "; ";
      }
    }
    
    // Record test result
    TestResult result;
    result.test_name = current_test_.name;
    result.passed = passed;
    result.duration = getNodeTime() - test_start_time_;
    result.details = result_details;
    result.actions_recorded = actions_recorded_;
    
    test_results_.push_back(result);
    
    LOG_INFO("Test %s: %s (%.2fs)", 
             current_test_.name.c_str(), 
             passed ? "PASSED" : "FAILED", 
             result.duration);
  }
  
  void finalizeTesting() {
    // Generate test report
    generateTestReport();
    
    // Publish results
    publishTestResults();
    
    LOG_INFO("All tests completed. %zu passed, %zu failed", 
             countPassedTests(), countFailedTests());
  }

  struct TestScenario {
    std::string name;
    std::string input;
    std::vector<std::string> expected_actions;
    std::function<void(TestEnvironment&)> environment_setup;
  };
  
  struct TestResult {
    std::string test_name;
    bool passed;
    double duration;
    std::string details;
    std::vector<std::string> actions_recorded;
  };
  
  // Test components
  std::vector<TestScenario> test_scenarios_;
  std::queue<size_t> test_queue_;
  TestScenario current_test_;
  std::vector<TestResult> test_results_;
  
  // Test state
  bool test_running_ = false;
  double test_start_time_ = 0.0;
  double test_timeout_ = 30.0;  // 30 seconds per test
  std::vector<std::string> actions_recorded_;
  
  // Test environment
  TestEnvironment test_env_;
  
  // Communication channels
  ISAAC_PROTO_RX(SystemActionsProto, system_actions);
  ISAAC_PROTO_TX(TestResultsProto, test_results);
};

}  // namespace testing
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::testing::VLASystemTester);
```

## Best Practices

### VLA System Design
1. **Modularity**: Keep vision, language, and action components modular for easy updates
2. **Robustness**: Handle uncertain or ambiguous inputs gracefully
3. **Real-time Performance**: Optimize for real-time response requirements
4. **Safety**: Include safety checks and fallback behaviors

### Conversational Design
1. **Natural Interaction**: Design conversations that feel natural to users
2. **Context Awareness**: Maintain context across conversation turns
3. **Error Recovery**: Handle misunderstandings gracefully
4. **Feedback**: Provide clear feedback about system state

### Integration Considerations
1. **Latency**: Minimize delays between speech input and action execution
2. **Reliability**: Ensure system operates reliably in real-world conditions
3. **Scalability**: Design to handle increasing complexity of tasks
4. **Maintainability**: Create systems that are easy to update and extend

## Summary

The Vision-Language-Action framework represents the cutting edge of conversational robotics, enabling natural human-robot interaction. By combining advanced perception, natural language understanding, and action execution, we can create robots that understand and respond to human commands in a natural way. The capstone project demonstrates how all these components work together to create an autonomous humanoid capable of complex tasks through voice commands.

## Exercises

1. Implement a simple VLA model that connects vision and language inputs to basic navigation actions.
2. Create a wake word detection system for activating the robot.
3. Design a dialogue manager for handling multi-turn conversations.
4. Build a complete task from voice command to action execution.