package ai.oyster.train.config

/**
 * Data class for all training configuration.
 * Fetched from server at registration time.
 *
 * @param modelName Name of the base model (e.g., "Qwen2.5-0.5B")
 * @param loraRank Rank of LoRA adaptation
 * @param loraAlpha Alpha parameter for LoRA scaling
 * @param localSteps Number of local training steps per round
 * @param batchSize Batch size for training
 * @param maxSeqLen Maximum sequence length for tokenization
 * @param serverAddress Address of the Flower server
 * @param compressionEnabled Whether to use gradient compression
 * @param topkRatio Ratio for top-k gradient compression (0.0-1.0)
 * @param syncIntervalSteps Steps between synchronizations with server
 */
data class TrainingConfig(
    val modelName: String,
    val loraRank: Int,
    val loraAlpha: Float,
    val localSteps: Int,
    val batchSize: Int,
    val maxSeqLen: Int,
    val serverAddress: String,
    val compressionEnabled: Boolean,
    val topkRatio: Float,
    val syncIntervalSteps: Int
)