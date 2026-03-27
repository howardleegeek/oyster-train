package ai.clawphones.train

import android.app.Application
import android.content.Context
import dagger.hilt.android.HiltAndroidApp

/**
 * Application class for Oyster Train federated learning app.
 * Configures Hilt dependency injection.
 */
@HiltAndroidApp
class OysterTrainApplication : Application() {

    companion object {
        lateinit var context: Context
    }

    override fun onCreate() {
        super.onCreate()
        context = applicationContext
    }
}
