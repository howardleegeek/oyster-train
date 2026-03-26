package ai.oyster.train

import android.util.Log
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject

/**
 * Bridge to the Python training engine via Chaquopy.
 * Replaces the old NativeEngine JNI approach.
 */
class TrainingEngine {

    private val py: Python = Python.getInstance()
    private val engine: PyObject = py.getModule("oyster.engine")

    companion object {
        private const val TAG = "TrainingEngine"
    }

    /**
     * Get device info (RAM, platform, PyTorch version).
     */
    suspend fun getDeviceInfo(): DeviceInfo = withContext(Dispatchers.IO) {
        val json = engine.callAttr("get_device_info").toString()
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

    /**
     * Run a quick model build + forward pass test.
     */
    suspend fun runQuickTest(): TestResult = withContext(Dispatchers.IO) {
        Log.i(TAG, "Running quick test...")
        val json = engine.callAttr("run_quick_test").toString()
        val obj = JSONObject(json)
        TestResult(
            success = obj.optString("status") == "ok",
            params = obj.optLong("params", 0),
            loss = obj.optDouble("loss", 0.0).toFloat(),
            torchVersion = obj.optString("torch_version"),
            error = obj.optString("error", ""),
        )
    }

    /**
     * Start federated training — connects to Flower server.
     */
    suspend fun startTraining(
        serverAddress: String,
        numRounds: Int = 100,
        localSteps: Int = 50
    ): Boolean = withContext(Dispatchers.IO) {
        Log.i(TAG, "Starting training: server=$serverAddress, rounds=$numRounds, steps=$localSteps")
        try {
            engine.callAttr("start_training", serverAddress, numRounds, localSteps)
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start training", e)
            false
        }
    }

    /**
     * Stop training gracefully.
     */
    suspend fun stopTraining() = withContext(Dispatchers.IO) {
        Log.i(TAG, "Stopping training")
        engine.callAttr("stop_training")
    }

    /**
     * Get current training status.
     */
    suspend fun getStatus(): TrainingStatus = withContext(Dispatchers.IO) {
        val json = engine.callAttr("get_status").toString()
        val obj = JSONObject(json)
        TrainingStatus(
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
    val params: Long,
    val loss: Float,
    val torchVersion: String,
    val error: String,
)

data class TrainingStatus(
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
)
