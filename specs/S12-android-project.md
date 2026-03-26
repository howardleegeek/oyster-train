---
task_id: S12-android-project
project: oyster-train
priority: 3
estimated_minutes: 45
depends_on: [S06]
modifies: ["client/android/"]
executor: glm
---
## Goal
Create a complete Android project structure around the existing 7 Kotlin files in client/android/. Must be buildable with Gradle and produce an APK.

## Constraints
- Package: `ai.clawphones.train`
- Min SDK: 28 (Android 9), Target SDK: 34 (Android 14)
- Existing Kotlin files to integrate:
  - FlowerClient.kt, FlowerService.kt, TrainingConfig.kt
  - ModelManager.kt, CompressionClient.kt, BatteryMonitor.kt, NetworkMonitor.kt
- Dependencies: Flower Android SDK, gRPC, Vulkan headers
- Architecture: MVVM with Jetpack Compose for UI
- Permissions: INTERNET, ACCESS_NETWORK_STATE, FOREGROUND_SERVICE, BATTERY_STATS, WAKE_LOCK

## Deliverables
- `client/android/app/build.gradle.kts` - Kotlin DSL build config
- `client/android/build.gradle.kts` - Root project config
- `client/android/settings.gradle.kts`
- `client/android/gradle.properties`
- `client/android/app/src/main/AndroidManifest.xml`
- `client/android/app/src/main/java/ai/clawphones/train/MainActivity.kt` - Basic Compose UI showing training status
- `client/android/app/src/main/java/ai/clawphones/train/ui/TrainingScreen.kt` - Training dashboard
- Move existing Kotlin files into proper package structure under `app/src/main/java/ai/clawphones/train/`
- `client/android/app/src/main/java/ai/clawphones/train/di/AppModule.kt` - Simple DI setup

## Do NOT
- Modify Python modules
- Add iOS or cross-platform code
- Include real API keys or server URLs (use BuildConfig fields)
