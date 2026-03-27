package ai.oyster.train

import android.content.Context

/**
 * JNI bridge to QVAC Fabric C++ library.
 * Provides methods to load models, train, and get/set LoRA weights.
 */
class NativeEngine private constructor(context: Context) {
    init {
        // Load the native library
        System.loadLibrary("qvac_fabric")
    }

    /**
     * Load a model with LoRA configuration.
     * @param modelPath Path to the model file
     * @param loraConfig LoRA configuration
     */
    external fun loadModel(modelPath: String, loraConfig: LoraConfig): Boolean

    /**
     * Perform a training step on the provided data.
     * @param data Input data as byte array
     * @param numSteps Number of training steps to perform
     * @return Training result containing loss and other metrics
     */
    external fun trainStep(data: ByteArray, numSteps: Int): TrainingResult

    /**
     * Get the current LoRA weights.
     * @return LoRA weights as byte array
     */
    external fun getLoraWeights(): ByteArray

    /**
     * Set the LoRA weights.
     * @param weights LoRA weights as byte array
     */
    external fun setLoraWeights(weights: ByteArray): Boolean

    /**
     * Get memory usage of the native engine.
     * @return Memory usage in bytes
     */
    external fun getMemoryUsage(): Long

    /**
     * Release the model and free native resources.
     */
    external fun releaseModel(): Boolean

    companion object {
        private var instance: NativeEngine? = null

        /**
         * Get the singleton instance of NativeEngine.
         * @param context Android context
         * @return NativeEngine instance
         */
        fun getInstance(context: Context): NativeEngine {
            return instance ?: synchronized(this) {
                instance ?: NativeEngine(context).also { instance = it }
            }
        }
    }
}

/**
 * Data class representing LoRA configuration.
 * @param rank Rank of LoRA
 * @param alpha Alpha parameter for LoRA scaling
 * @param targetModules List of module names to apply LoRA to
 */
data class LoraConfig(
    val rank: Int,
    val alpha: Float,
    val targetModules: List<String>
)

/**
 * Data class representing the result of a training step.
 * @param loss Loss value from the training step
 * @param additionalMetrics Additional metrics from training (optional)
 */
data class TrainingResult(
    val loss: Float,
    val additionalMetrics: Map<String, Float> = emptyMap()
)