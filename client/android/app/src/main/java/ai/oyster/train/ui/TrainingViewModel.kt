package ai.oyster.train.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import ai.oyster.train.TrainingEngine
import ai.oyster.train.TrainingStatus
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class TrainingUiState(
    val isTrainingActive: Boolean = false,
    val isConnectedToServer: Boolean = false,
    val serverAddress: String = "Not connected",
    val currentRound: Int = 0,
    val totalRounds: Int = 100,
    val localSteps: Int = 0,
    val totalLocalSteps: Int = 50,
    val currentLoss: Float = 0.0f,
    val accuracy: Float = 0.0f,
    val totalStepsCompleted: Int = 0,
    val memoryUsageMB: Int = 0,
    val uptime: String = "0h 0m",
    val modelName: String = "LeWM (JEPA)",
    val loraRank: Int = 0,  // 0 = full parameter training
    val paramsM: Float = 0f,
    val torchVersion: String = "",
    val modelTestPassed: Boolean = false,
    val errorMessage: String = "",
)

/**
 * ViewModel that connects the Compose UI to the real Python training engine.
 * No more simulation — this drives actual PyTorch + Flower FL.
 */
class TrainingViewModel : ViewModel() {

    private val engine = TrainingEngine()
    private val _uiState = MutableStateFlow(TrainingUiState())
    val uiState: StateFlow<TrainingUiState> = _uiState.asStateFlow()

    private var startTime = 0L

    init {
        // Run quick model test on startup
        viewModelScope.launch {
            try {
                val result = engine.runQuickTest()
                _uiState.value = _uiState.value.copy(
                    modelTestPassed = result.success,
                    paramsM = result.params / 1_000_000f,
                    torchVersion = result.torchVersion,
                    currentLoss = result.loss,
                    errorMessage = result.error,
                    modelName = "LeWM ${String.format("%.1f", result.params / 1_000_000f)}M",
                )
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    errorMessage = "Model init failed: ${e.message}"
                )
            }
        }
    }

    fun startTraining(serverAddress: String = "10.0.2.2:8080") {
        viewModelScope.launch {
            startTime = System.currentTimeMillis()

            val success = engine.startTraining(
                serverAddress = serverAddress,
                numRounds = 100,
                localSteps = 50,
            )

            if (success) {
                _uiState.value = _uiState.value.copy(
                    isTrainingActive = true,
                    isConnectedToServer = true,
                    serverAddress = serverAddress,
                    errorMessage = "",
                )
                // Start polling status
                pollStatus()
            } else {
                _uiState.value = _uiState.value.copy(
                    errorMessage = "Failed to connect to $serverAddress"
                )
            }
        }
    }

    fun stopTraining() {
        viewModelScope.launch {
            engine.stopTraining()
            _uiState.value = _uiState.value.copy(
                isTrainingActive = false
            )
        }
    }

    private fun pollStatus() {
        viewModelScope.launch {
            while (_uiState.value.isTrainingActive) {
                delay(1000)
                try {
                    val status = engine.getStatus()
                    val elapsed = (System.currentTimeMillis() - startTime) / 1000
                    val hours = elapsed / 3600
                    val minutes = (elapsed % 3600) / 60

                    _uiState.value = _uiState.value.copy(
                        currentRound = status.round,
                        totalRounds = if (status.totalRounds > 0) status.totalRounds else _uiState.value.totalRounds,
                        localSteps = status.step,
                        totalLocalSteps = if (status.totalSteps > 0) status.totalSteps else _uiState.value.totalLocalSteps,
                        currentLoss = status.loss,
                        memoryUsageMB = status.memoryMb,
                        totalStepsCompleted = _uiState.value.totalStepsCompleted + 1,
                        uptime = "${hours}h ${minutes}m",
                        isTrainingActive = status.state == "training" || status.state == "connecting",
                        isConnectedToServer = status.state != "error",
                        errorMessage = status.error,
                    )

                    // Auto-stop if training finished or errored
                    if (status.state == "idle" || status.state == "error") {
                        _uiState.value = _uiState.value.copy(isTrainingActive = false)
                    }
                } catch (e: Exception) {
                    // Keep polling, engine might recover
                }
            }
        }
    }
}
