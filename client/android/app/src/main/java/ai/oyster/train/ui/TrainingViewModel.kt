package ai.oyster.train.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import ai.oyster.train.ModelType
import ai.oyster.train.TrainingEngine
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class ModelStatus(
    val name: String = "",
    val ready: Boolean = false,
    val training: Boolean = false,
    val round: Int = 0,
    val totalRounds: Int = 100,
    val step: Int = 0,
    val totalSteps: Int = 0,
    val loss: Float = 0f,
    val paramsM: Float = 0f,
    val tokensPerSec: Float = 0f,
    val error: String = "",
)

data class TrainingUiState(
    val selectedModel: ModelType = ModelType.BOTH,
    val serverAddress: String = "10.0.2.2:8080",
    val isTrainingActive: Boolean = false,
    val lewm: ModelStatus = ModelStatus(name = "LeWM JEPA (4.8M)"),
    val qwen: ModelStatus = ModelStatus(name = "Qwen2.5-0.5B"),
    val qwenDownloaded: Boolean = false,
    val qwenDownloading: Boolean = false,
    val qwenDownloadProgress: String = "",
    val uptime: String = "0h 0m",
    val torchVersion: String = "",
    val errorMessage: String = "",
)

class TrainingViewModel : ViewModel() {

    private val engine = TrainingEngine()
    private val _uiState = MutableStateFlow(TrainingUiState())
    val uiState: StateFlow<TrainingUiState> = _uiState.asStateFlow()

    private var startTime = 0L

    init {
        // Test LeWM on startup
        viewModelScope.launch {
            try {
                val result = engine.testLeWM()
                _uiState.value = _uiState.value.copy(
                    lewm = _uiState.value.lewm.copy(
                        ready = result.success,
                        paramsM = result.params / 1_000_000f,
                        loss = result.loss,
                        error = result.error,
                    ),
                    torchVersion = result.torchVersion,
                )
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    errorMessage = "LeWM init failed: ${e.message}"
                )
            }
        }
    }

    fun selectModel(model: ModelType) {
        _uiState.value = _uiState.value.copy(selectedModel = model)
    }

    fun downloadQwen() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(qwenDownloading = true, qwenDownloadProgress = "Downloading ~350MB...")
            try {
                val result = engine.downloadQwen()
                if (result.success) {
                    _uiState.value = _uiState.value.copy(
                        qwenDownloading = false,
                        qwenDownloadProgress = "Downloaded ${result.sizeMb}MB",
                    )
                    // Load model after download
                    loadQwen()
                } else {
                    _uiState.value = _uiState.value.copy(
                        qwenDownloading = false,
                        errorMessage = "Download failed: ${result.error}",
                    )
                }
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    qwenDownloading = false,
                    errorMessage = "Download error: ${e.message}",
                )
            }
        }
    }

    private suspend fun loadQwen() {
        val result = engine.loadQwen()
        _uiState.value = _uiState.value.copy(
            qwenDownloaded = result.success,
            qwen = _uiState.value.qwen.copy(
                ready = result.success,
                tokensPerSec = result.tokensPerSec,
                error = result.error,
            ),
        )
    }

    fun startTraining(serverAddress: String = "10.0.2.2:8080") {
        viewModelScope.launch {
            startTime = System.currentTimeMillis()
            val model = _uiState.value.selectedModel

            _uiState.value = _uiState.value.copy(
                serverAddress = serverAddress,
                errorMessage = "",
            )

            when (model) {
                ModelType.LEWM -> {
                    val ok = engine.startLeWMTraining(serverAddress)
                    _uiState.value = _uiState.value.copy(
                        isTrainingActive = ok,
                        lewm = _uiState.value.lewm.copy(training = ok),
                    )
                }
                ModelType.QWEN -> {
                    val ok = engine.startQwenTraining(serverAddress)
                    _uiState.value = _uiState.value.copy(
                        isTrainingActive = ok,
                        qwen = _uiState.value.qwen.copy(training = ok),
                    )
                }
                ModelType.BOTH -> {
                    val (lewmOk, qwenOk) = engine.startDualTraining(serverAddress)
                    _uiState.value = _uiState.value.copy(
                        isTrainingActive = lewmOk || qwenOk,
                        lewm = _uiState.value.lewm.copy(training = lewmOk),
                        qwen = _uiState.value.qwen.copy(training = qwenOk),
                    )
                }
            }

            if (_uiState.value.isTrainingActive) pollStatus()
        }
    }

    fun stopTraining() {
        viewModelScope.launch {
            engine.stopAll()
            _uiState.value = _uiState.value.copy(
                isTrainingActive = false,
                lewm = _uiState.value.lewm.copy(training = false),
                qwen = _uiState.value.qwen.copy(training = false),
            )
        }
    }

    private fun pollStatus() {
        viewModelScope.launch {
            while (_uiState.value.isTrainingActive) {
                delay(1000)
                try {
                    val elapsed = (System.currentTimeMillis() - startTime) / 1000
                    val uptime = "${elapsed / 3600}h ${(elapsed % 3600) / 60}m"

                    if (_uiState.value.lewm.training) {
                        val s = engine.getLeWMStatus()
                        _uiState.value = _uiState.value.copy(
                            uptime = uptime,
                            lewm = _uiState.value.lewm.copy(
                                round = s.round, totalRounds = s.totalRounds,
                                step = s.step, totalSteps = s.totalSteps,
                                loss = s.loss,
                                training = s.state == "training" || s.state == "connecting",
                            ),
                        )
                    }

                    if (_uiState.value.qwen.training) {
                        val s = engine.getQwenStatus()
                        _uiState.value = _uiState.value.copy(
                            uptime = uptime,
                            qwen = _uiState.value.qwen.copy(
                                round = s.round, totalRounds = s.totalRounds,
                                step = s.step, totalSteps = s.totalSteps,
                                loss = s.loss, tokensPerSec = s.tokensPerSec,
                                training = s.state == "training" || s.state == "connecting",
                            ),
                        )
                    }

                    // Auto-stop if both finished
                    if (!_uiState.value.lewm.training && !_uiState.value.qwen.training) {
                        _uiState.value = _uiState.value.copy(isTrainingActive = false)
                    }
                } catch (_: Exception) {}
            }
        }
    }
}
