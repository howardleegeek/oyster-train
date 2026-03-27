package ai.clawphones.train.di

import ai.clawphones.train.BuildConfig
import ai.clawphones.train.DataManager
import ai.clawphones.train.FlowerClient
import ai.clawphones.train.NativeEngine
import ai.clawphones.train.OysterTrainApplication
import ai.clawphones.train.config.TrainingConfig
import android.content.Context
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

/**
 * Hilt dependency injection module for the Oyster Train app.
 * Provides singleton instances of core components.
 */
@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    /**
     * Provides the training configuration.
     * In a real implementation, this would be fetched from the server.
     */
    @Provides
    @Singleton
    fun provideTrainingConfig(): TrainingConfig {
        return TrainingConfig(
            modelName = "Qwen2.5-0.5B",
            loraRank = 8,
            loraAlpha = 16.0f,
            localSteps = 50,
            batchSize = 4,
            maxSeqLen = 512,
            serverAddress = BuildConfig.FLOWER_SERVER_URL,
            compressionEnabled = true,
            topkRatio = 0.1f,
            syncIntervalSteps = 50
        )
    }

    /**
     * Provides the NativeEngine singleton.
     */
    @Provides
    @Singleton
    fun provideNativeEngine(@ApplicationContext context: Context): NativeEngine {
        return NativeEngine.getInstance(context)
    }

    /**
     * Provides the DataManager singleton.
     */
    @Provides
    @Singleton
    fun provideDataManager(@ApplicationContext context: Context): DataManager {
        return DataManager.getInstance(context)
    }

    /**
     * Provides the FlowerClient singleton.
     */
    @Provides
    @Singleton
    fun provideFlowerClient(
        nativeEngine: NativeEngine,
        trainingConfig: TrainingConfig
    ): FlowerClient {
        return FlowerClient(
            serverAddress = trainingConfig.serverAddress,
            nativeEngine = nativeEngine,
            trainingConfig = trainingConfig
        )
    }
}
