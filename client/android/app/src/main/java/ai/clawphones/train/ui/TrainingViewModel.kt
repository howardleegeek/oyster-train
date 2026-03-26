package ai.clawphones.train.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * UI state for the TrainingScreen.
 */
data class TrainingUiState(
    val isTrainingActive: Boolean = false,
    val isConnectedToServer: Boolean = false,
    val serverAddress: String = "Not connected",
    val currentRound: Int = 0,
    val totalRounds: Int = 100,
    val localSteps: Int = 0,
    val totalLocalSteps: Int = 50,
    val currentLoss: Float = 1.0f,
    val accuracy: Float = 0.0f,
    val totalStepsCompleted: Int = 0,
    val memoryUsageMB: Int = 0,
    val uptime: String = "0h 0m",
    val modelName: String = "Loading...",
    val loraRank: Int = 0
)

/**
 * ViewModel for the TrainingScreen.
 * Manages training state and UI updates.
 */
@HiltViewModel
class TrainingViewModel @Inject constructor() : ViewModel() {

    private val _uiState = MutableStateFlow(TrainingUiState())
    val uiState: StateFlow<TrainingUiState> = _uiState.asStateFlow()

    fun startTraining() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(
                isTrainingActive = true,
                isConnectedToServer = true,
                serverAddress = "192.168.1.100:50051",
                modelName = "Qwen2.5-0.5B",
                loraRank = 8
            )

            // Simulate training progress
            simulateTraining()
        }
    }

    fun stopTraining() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(
                isTrainingActive = false
            )
        }
    }

    private suspend fun simulateTraining() {
        while (_uiState.value.isTrainingActive) {
            kotlinx.coroutines.delay(2000)

            val currentState = _uiState.value

            // Simulate progress
            val newRound = if (currentState.currentRound < currentState.totalRounds) {
                if (currentState.localSteps >= currentState.totalLocalSteps) {
                    currentState.currentRound + 1
                } else {
                    currentState.currentRound
                }
            } else {
                currentState.currentRound
            }

            val newLocalSteps = if (currentState.localSteps < currentState.totalLocalSteps) {
                currentState.localSteps + 1
            } else {
                0
            }

            val newLoss = 1.0f - (currentState.currentRound.toFloat() / currentState.totalRounds * 0.8f)
            val newAccuracy = currentState.currentRound.toFloat() / currentState.totalRounds * 0.95f

            _uiState.value = currentState.copy(
                currentRound = newRound,
                localSteps = newLocalSteps,
                currentLoss = newLoss.coerceIn(0.0f, 1.0f),
                accuracy = newAccuracy.coerceIn(0.0f, 1.0f),
                totalStepsCompleted = currentState.totalStepsCompleted + 1,
                memoryUsageMB = (50 + (currentState.currentRound * 2)).coerceAtMost(200)
            )
        }
    }
}
