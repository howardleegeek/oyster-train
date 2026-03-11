# Oyster Phone Training - Android Client

This is the Android client application for the Oyster Phone Training system. The app runs as a background service on Android devices, managing local LoRA training and communication with a Flower federated learning server.

## Architecture

The client consists of several key components:

1. **TrainingService** (`TrainingService.kt`) - Android foreground service that manages the training lifecycle
   - Checks preconditions: charging + WiFi connected + screen off
   - Starts/stops training based on conditions
   - Shows persistent notification with training progress
   - Handles battery optimization

2. **FlowerClient** (`FlowerClient.kt`) - Flower federated learning gRPC client
   - Implements Flower NumPyClient protocol via gRPC
   - Communicates with Flower server for federated learning rounds
   - Handles parameter exchange and training/evaluation

3. **NativeEngine** (`NativeEngine.kt`) - JNI bridge to QVAC Fabric C++ library
   - Interface to the native training engine
   - Loads models, performs training steps, manages LoRA weights

4. **DataManager** (`DataManager.kt`) - Local training data management
   - Manages storage and tokenization of training data
   - Prepares batches for training
   - Tracks data statistics

5. **TrainingConfig** (`config/TrainingConfig.kt`) - Configuration data class
   - Holds all training parameters fetched from server

6. **BootReceiver** (`BootReceiver.kt`) - Starts service on device boot

## Building

To build this Android client, you'll need:
- Android Studio Arctic Fox or later
- Android SDK 34 (API level 34)
- NDK version compatible with CMake 3.22.1

The build is configured for:
- minSdk 30 (Android 8.1)
- targetSdk 34 (Android 14)
- NDK abiFilters: arm64-v8a (for Unisoc T616)

## Permissions

The app requires the following permissions:
- `INTERNET` - For gRPC communication with Flower server
- `ACCESS_NETWORK_STATE` - To check WiFi connectivity
- `FOREGROUND_SERVICE` - Required for long-running training service
- `RECEIVE_BOOT_COMPLETED` - To start service after device reboot
- `REQUEST_IGNORE_BATTERY_OPTIMIZATIONS` - To request exemption from battery optimizations

## Privacy

This client is designed with privacy in mind:
- Training data never leaves the device
- Only compressed gradient updates are sent to the server
- All data is stored in the app's private internal storage

## Battery Efficiency

Training only occurs when:
- Device is charging
- Connected to WiFi
- Screen is off

This ensures minimal impact on user experience and battery life.