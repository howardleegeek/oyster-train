plugins {
    id("com.android.application")
    id("kotlin-android")
}

android {
    compileSdk = 34

    defaultConfig {
        applicationId = "ai.oyster.train"
        minSdk = 30
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        
        // NDK configuration for JNI (arm64-v8a for Unisoc T616)
        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17 -frtti -fexceptions"
                abiFilters += "arm64-v8a"
            }
        }
    }

    buildTypes {
        release {
            minifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    
    // Specify that we're using CMake for native builds
    externalNativeBuild {
        cmake {
            path = "src/main/cpp/CMakeLists.txt"
            version = "3.22.1"
        }
    }
}

dependencies {
    implementation("org.jetbrains.kotlin:kotlin-stdlib:1.8.0")
    implementation("androidx.core:core-ktx:1.9.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.8.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    
    // gRPC dependencies
    implementation("io.grpc:grpc-netty-shaded:1.57.0")
    implementation("io.grpc:grpc-protobuf:1.57.0")
    implementation("io.grpc:grpc-stub:1.57.0")
    
    // Protobuf dependencies
    implementation("com.google.protobuf:protobuf-lite:3.21.12")
    
    // Flower dependencies (if available as Android SDK)
    // implementation("io.flwr:flwr-android:0.0.1") // Placeholder
    
    // Lifecycle dependencies
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.6.2")
    implementation("androidx.lifecycle:lifecycle-service:2.6.2")
    
    // WorkManager for periodic tasks (alternative to foreground service)
    implementation("androidx.work:work-runtime-ktx:2.7.1")
    
    // Testing dependencies
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
}