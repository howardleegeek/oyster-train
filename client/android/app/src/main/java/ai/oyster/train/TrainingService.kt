package ai.oyster.train

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.BatteryManager
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat

/**
 * Foreground service that manages the training lifecycle.
 * Training only occurs when: charging + WiFi connected + screen off.
 */
class TrainingService : Service() {

    private lateinit var notificationManager: NotificationManager
    private lateinit var powerManager: PowerManager
    private lateinit var connectivityManager: ConnectivityManager
    private lateinit var batteryManager: BatteryManager

    private var isTrainingActive = false

    override fun onCreate() {
        super.onCreate()
        notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        connectivityManager = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        batteryManager = getSystemService(Context.BATTERY_SERVICE) as BatteryManager

        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // Start monitoring conditions
        startMonitoring()
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? {
        // Not used for this service
        return null
    }

    override fun onDestroy() {
        super.onDestroy()
        stopMonitoring()
    }

    private fun startMonitoring() {
        // Start a background thread to check conditions periodically
        Thread {
            while (true) {
                if (shouldTrain()) {
                    if (!isTrainingActive) {
                        startTraining()
                    }
                } else {
                    if (isTrainingActive) {
                        stopTraining()
                    }
                }
                // Check every 30 seconds
                Thread.sleep(30_000)
            }
        }.start()
    }

    private fun stopMonitoring() {
        // In a real implementation, we would stop the monitoring thread
        // For simplicity, we rely on the service being destroyed
    }

    private fun shouldTrain(): Boolean {
        return isCharging() && isWifiConnected() && isScreenOff()
    }

    private fun isCharging(): Boolean {
        val intent = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val status = intent?.getIntExtra(BatteryManager.EXTRA_STATUS, -1)
        return status == BatteryManager.BATTERY_STATUS_CHARGING ||
                status == BatteryManager.BATTERY_STATUS_FULL
    }

    private fun isWifiConnected(): Boolean {
        val network = connectivityManager.activeNetwork
        if (network == null) return false
        val capabilities = connectivityManager.getNetworkCapabilities(network)
        return capabilities?.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) == true
    }

    private fun isScreenOff(): Boolean {
        return !powerManager.isInteractive
    }

    private fun startTraining() {
        isTrainingActive = true
        // Notify the user via notification that training has started
        updateNotification("Training in progress...", R.drawable.ic_training)
        // Here we would start the actual training process
        // For example, by calling into the NativeEngine and FlowerClient
    }

    private fun stopTraining() {
        isTrainingActive = false
        updateNotification("Training paused", R.drawable.ic_paused)
        // Stop the training process
    }

    private fun updateNotification(text: String, icon: Int) {
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Oyster Training")
            .setContentText(text)
            .setSmallIcon(icon)
            .setOngoing(true)
            .build()

        notificationManager.notify(NOTIFICATION_ID, notification)
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val name = "Oyster Training Channel"
            val descriptionText = "Channel for Oyster training service"
            val importance = NotificationManager.IMPORTANCE_LOW
            val channel = NotificationChannel(CHANNEL_ID, name, importance).apply {
                description = descriptionText
            }
            notificationManager.createNotificationChannel(channel)
        }
    }

    companion object {
        private const val CHANNEL_ID = "oyster_training_channel"
        private const val NOTIFICATION_ID = 1
    }
}