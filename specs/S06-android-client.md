---
task_id: S06-android-client
project: oyster-train
priority: 2
estimated_minutes: 50
depends_on: []
modifies: ["client/android/"]
executor: glm
---

## Goal
Create Android client app structure for Oyster Phone Training. This app runs as a background service on UBS1 phones, managing local LoRA training and Flower server communication.

## Context
- Target: Android 14 (API 34), Unisoc T616, 6GB RAM
- App runs as Android foreground service (required for long-running training)
- Training only happens when: charging + WiFi connected + screen off
- Communication with Flower server via gRPC
- Native training engine: QVAC Fabric (C++ via JNI)

## Deliverables

### client/android/app/src/main/java/ai/oyster/train/TrainingService.kt
- Android Foreground Service that manages training lifecycle
- Check preconditions: isCharging(), isWifiConnected(), isScreenOff()
- Start/stop training based on conditions
- Show persistent notification with training progress
- Handle battery optimization (request exemption from Doze)

### client/android/app/src/main/java/ai/oyster/train/FlowerClient.kt
- Implement Flower NumPyClient protocol via gRPC
- `getParameters()`: Read LoRA weights from native engine
- `fit(parameters, config)`:
  1. Apply global LoRA params to native engine
  2. Trigger local training for N steps
  3. Extract LoRA delta
  4. Compress using compression pipeline
  5. Return compressed update + metrics
- `evaluate(parameters, config)`: Run evaluation, return loss
- gRPC connection to configurable server address

### client/android/app/src/main/java/ai/oyster/train/NativeEngine.kt
- JNI bridge to QVAC Fabric C++ library
- Methods:
  - `loadModel(modelPath: String, loraConfig: LoraConfig)`
  - `trainStep(data: ByteArray, numSteps: Int): TrainingResult`
  - `getLoraWeights(): ByteArray`
  - `setLoraWeights(weights: ByteArray)`
  - `getMemoryUsage(): Long`
  - `releaseModel()`

### client/android/app/src/main/java/ai/oyster/train/DataManager.kt
- Manage local training data on device
- Store data in app's internal storage (privacy)
- Tokenize text data using Qwen2.5 tokenizer (SentencePiece)
- Batch data for training (batch_size=4, max_seq_len=256)
- Track data statistics (num_samples, distribution)

### client/android/app/src/main/java/ai/oyster/train/config/TrainingConfig.kt
- Kotlin data class for all training configuration
- Fetched from server at registration time
- Fields: model_name, lora_rank, lora_alpha, local_steps, batch_size, max_seq_len,
  server_address, compression_enabled, topk_ratio, sync_interval_steps

### client/android/app/src/main/AndroidManifest.xml
- Permissions: INTERNET, ACCESS_NETWORK_STATE, FOREGROUND_SERVICE,
  RECEIVE_BOOT_COMPLETED, REQUEST_IGNORE_BATTERY_OPTIMIZATIONS
- Service declaration for TrainingService
- BroadcastReceiver for BOOT_COMPLETED (auto-start)
- Application metadata

### client/android/build.gradle.kts
- Android Gradle build config
- minSdk = 30, targetSdk = 34, compileSdk = 34
- Dependencies: grpc-android, protobuf-lite, flwr-android (if available)
- NDK configuration for JNI (abiFilters = arm64-v8a)
- ProGuard rules

## Constraints
- This is the APP STRUCTURE and Kotlin code, not compilable without Android SDK
- Focus on correct architecture and complete Kotlin code
- All classes must be well-documented with KDoc comments
- Follow Android best practices (foreground service, WorkManager consideration)
- Privacy: no user data leaves the device except compressed gradients
- Battery-conscious: only train when conditions met

## Acceptance Criteria
- [ ] All Kotlin files have correct syntax and imports
- [ ] AndroidManifest.xml is complete with all permissions and components
- [ ] build.gradle.kts has correct dependencies and NDK config
- [ ] TrainingService handles all lifecycle states correctly
- [ ] FlowerClient implements full FL protocol
- [ ] README.md explains the architecture
