package ai.clawphones.train

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import ai.clawphones.train.ui.TrainingScreen
import ai.clawphones.train.ui.theme.OysterTrainTheme
import dagger.hilt.android.AndroidEntryPoint

/**
 * Main activity for the Oyster Train federated learning app.
 * Uses Jetpack Compose for UI.
 */
@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            OysterTrainTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    OysterTrainNavGraph()
                }
            }
        }
    }
}

@Composable
fun OysterTrainNavGraph() {
    val navController = rememberNavController()

    NavHost(
        navController = navController,
        startDestination = "training"
    ) {
        composable("training") {
            TrainingScreen()
        }
    }
}
