package ai.oyster.train

import android.util.Log
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject

enum class ModelType { LEWM, QWEN, BOTH }

/**
 * Dual-model training engine — LeWM (world model) + Qwen (LLM).
 * Bridges Kotlin UI to Python engines via Chaquopy.
 */
class TrainingEngine {

    private val py: Python = Python.getInstance()
    private val lewmEngine: PyObject = py.getModule("oyster.engine")
    private val qwenEngine: PyObject = py.getModule("oyster.qwen_engine")

    companion object {
        private const val TAG = "TrainingEngine"
    }

    // ─── Device info ─────────────────────────────────────────────

    suspend fun getDeviceInfo(): DeviceInfo = withContext(Dispatchers.IO) {
        val json = lewmEngine.callAttr("get_device_info").toString()
        val obj = JSONObject(json)
        DeviceInfo(
            platform = obj.optString("platform"),
            machine = obj.optString("machine"),
            pythonVersion = obj.optString("python"),
            torchVersion = obj.optString("torch"),
            ramGb = obj.optDouble("ram_gb", 0.0).toFloat(),
            isArm = obj.optBoolean("is_arm", false),
        )
    }

    // ─── LeWM (world model) ──────────────────────────────────────

    suspend fun testLeWM(): TestResult = withContext(Dispatchers.IO) {
        Log.i(TAG, "Testing LeWM...")
        val json = lewmEngine.callAttr("run_quick_test").toString()
        val obj = JSONObject(json)
        TestResult(
            success = obj.optString("status") == "ok",
            modelName = "LeWM JEPA",
            params = obj.optLong("params", 0),
            loss = obj.optDouble("loss", 0.0).toFloat(),
            torchVersion = obj.optString("torch_version"),
            error = obj.optString("error", ""),
            tokensPerSec = 0f,
        )
    }

    suspend fun startLeWMTraining(serverAddress: String): Boolean = withContext(Dispatchers.IO) {
        Log.i(TAG, "Starting LeWM training: server=$serverAddress")
        try {
            lewmEngine.callAttr("start_training", serverAddress, 100, 50)
            true
        } catch (e: Exception) {
            Log.e(TAG, "LeWM training failed", e)
            false
        }
    }

    suspend fun getLeWMStatus(): TrainingStatus = withContext(Dispatchers.IO) {
        val json = lewmEngine.callAttr("get_status").toString()
        parseStatus(json, "LeWM")
    }

    suspend fun stopLeWM() = withContext(Dispatchers.IO) {
        lewmEngine.callAttr("stop_training")
    }

    // ─── Qwen (LLM) ─────────────────────────────────────────────

    suspend fun downloadQwen(): DownloadResult = withContext(Dispatchers.IO) {
        Log.i(TAG, "Downloading Qwen2.5-0.5B GGUF...")
        val json = qwenEngine.callAttr("download_model").toString()
        val obj = JSONObject(json)
        DownloadResult(
            success = obj.optString("status") != "error",
            sizeMb = obj.optDouble("size_mb", 0.0).toFloat(),
            error = obj.optString("error", ""),
        )
    }

    suspend fun loadQwen(modelDir: String? = null): TestResult = withContext(Dispatchers.IO) {
        Log.i(TAG, "Loading Qwen model...")
        if (modelDir != null) {
            qwenEngine.callAttr("set_model_dir", modelDir)
        }
        val loadJson = qwenEngine.callAttr("load_model", 512, 4).toString()
        val loadObj = JSONObject(loadJson)
        if (loadObj.optString("status") == "error") {
            return@withContext TestResult(
                success = false, modelName = "Qwen2.5-0.5B", params = 0,
                loss = 0f, torchVersion = "", tokensPerSec = 0f,
                error = loadObj.optString("error"),
            )
        }

        // Run quick inference test
        val testJson = qwenEngine.callAttr("run_quick_test").toString()
        val obj = JSONObject(testJson)
        TestResult(
            success = obj.optString("status") == "ok",
            modelName = obj.optString("model", "Qwen2.5-0.5B"),
            params = obj.optLong("lora_params", 0),
            loss = 0f,
            torchVersion = "",
            tokensPerSec = obj.optDouble("tokens_per_sec", 0.0).toFloat(),
            error = obj.optString("error", ""),
        )
    }

    suspend fun generateText(prompt: String, maxTokens: Int = 64): GenerateResult = withContext(Dispatchers.IO) {
        val json = qwenEngine.callAttr("generate", prompt, maxTokens).toString()
        val obj = JSONObject(json)
        GenerateResult(
            text = obj.optString("text", ""),
            tokensPerSec = obj.optDouble("tokens_per_sec", 0.0).toFloat(),
            error = obj.optString("error", ""),
        )
    }

    suspend fun startQwenTraining(serverAddress: String): Boolean = withContext(Dispatchers.IO) {
        Log.i(TAG, "Starting Qwen LoRA training: server=$serverAddress")
        try {
            qwenEngine.callAttr("start_training", serverAddress, 100, 10)
            true
        } catch (e: Exception) {
            Log.e(TAG, "Qwen training failed", e)
            false
        }
    }

    suspend fun getQwenStatus(): TrainingStatus = withContext(Dispatchers.IO) {
        val json = qwenEngine.callAttr("get_status").toString()
        parseStatus(json, "Qwen")
    }

    suspend fun stopQwen() = withContext(Dispatchers.IO) {
        qwenEngine.callAttr("stop_training")
    }

    // ─── Dual model helpers ──────────────────────────────────────

    suspend fun startDualTraining(serverAddress: String): Pair<Boolean, Boolean> {
        val lewm = startLeWMTraining(serverAddress)
        val qwen = startQwenTraining(serverAddress)
        return Pair(lewm, qwen)
    }

    suspend fun stopAll() {
        stopLeWM()
        stopQwen()
    }

    // ─── Private ─────────────────────────────────────────────────

    private fun parseStatus(json: String, model: String): TrainingStatus {
        val obj = JSONObject(json)
        return TrainingStatus(
            model = model,
            state = obj.optString("state", "idle"),
            round = obj.optInt("round", 0),
            totalRounds = obj.optInt("total_rounds", 0),
            step = obj.optInt("step", 0),
            totalSteps = obj.optInt("total_steps", 0),
            loss = obj.optDouble("loss", 0.0).toFloat(),
            paramsM = obj.optDouble("params_m", 0.0).toFloat(),
            memoryMb = obj.optInt("memory_mb", 0),
            server = obj.optString("server", ""),
            error = obj.optString("error", ""),
            tokensPerSec = obj.optDouble("tokens_per_sec", 0.0).toFloat(),
            loraSizeMb = obj.optDouble("lora_size_mb", 0.0).toFloat(),
        )
    }
}

data class DeviceInfo(
    val platform: String,
    val machine: String,
    val pythonVersion: String,
    val torchVersion: String,
    val ramGb: Float,
    val isArm: Boolean,
)

data class TestResult(
    val success: Boolean,
    val modelName: String = "",
    val params: Long,
    val loss: Float,
    val torchVersion: String,
    val tokensPerSec: Float = 0f,
    val error: String,
)

data class DownloadResult(
    val success: Boolean,
    val sizeMb: Float,
    val error: String,
)

data class GenerateResult(
    val text: String,
    val tokensPerSec: Float,
    val error: String,
)

data class TrainingStatus(
    val model: String = "",
    val state: String,
    val round: Int,
    val totalRounds: Int,
    val step: Int,
    val totalSteps: Int,
    val loss: Float,
    val paramsM: Float,
    val memoryMb: Int,
    val server: String,
    val error: String,
    val tokensPerSec: Float = 0f,
    val loraSizeMb: Float = 0f,
)
