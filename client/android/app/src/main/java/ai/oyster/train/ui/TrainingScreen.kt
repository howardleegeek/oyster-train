package ai.oyster.train.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TrainingScreen(viewModel: TrainingViewModel = viewModel()) {
    val uiState by viewModel.uiState.collectAsState()
    var serverInput by remember { mutableStateOf("10.0.2.2:8080") }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Oyster Train") },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    titleContentColor = MaterialTheme.colorScheme.onPrimaryContainer,
                )
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            // Status indicator
            Row(verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Box(Modifier.size(16.dp).background(
                    if (uiState.isTrainingActive) Color(0xFF4CAF50) else Color(0xFF9E9E9E),
                    CircleShape
                ))
                Text(
                    if (uiState.isTrainingActive) "Training Active" else "Idle",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                )
            }

            Spacer(Modifier.height(16.dp))

            // Model info
            if (uiState.modelTestPassed) {
                Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(
                    containerColor = Color(0xFFE8F5E9))) {
                    Column(Modifier.padding(16.dp)) {
                        Text("Model Ready", fontWeight = FontWeight.Bold,
                            color = Color(0xFF2E7D32))
                        Text("${uiState.modelName} | PyTorch ${uiState.torchVersion}",
                            style = MaterialTheme.typography.bodySmall)
                    }
                }
            } else if (uiState.errorMessage.isNotEmpty()) {
                Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(
                    containerColor = Color(0xFFFFEBEE))) {
                    Text(uiState.errorMessage, Modifier.padding(16.dp), color = Color(0xFFC62828))
                }
            }

            Spacer(Modifier.height(16.dp))

            // Server address input
            OutlinedTextField(
                value = serverInput,
                onValueChange = { serverInput = it },
                label = { Text("Flower Server (ip:port)") },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
                enabled = !uiState.isTrainingActive,
            )

            Spacer(Modifier.height(16.dp))

            // Start / Stop buttons
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                Button(
                    onClick = { viewModel.startTraining(serverInput) },
                    enabled = !uiState.isTrainingActive && uiState.modelTestPassed,
                    modifier = Modifier.weight(1f),
                    shape = RoundedCornerShape(8.dp),
                ) { Text("Join Network") }

                Button(
                    onClick = { viewModel.stopTraining() },
                    enabled = uiState.isTrainingActive,
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error),
                    shape = RoundedCornerShape(8.dp),
                ) { Text("Stop") }
            }

            Spacer(Modifier.height(24.dp))

            // Training stats
            Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
                Column(Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
                    Text("Training Statistics", fontWeight = FontWeight.Bold,
                        style = MaterialTheme.typography.titleMedium)

                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                        StatItem("Round", "${uiState.currentRound}/${uiState.totalRounds}")
                        StatItem("Loss", String.format("%.4f", uiState.currentLoss))
                        StatItem("Steps", "${uiState.localSteps}/${uiState.totalLocalSteps}")
                    }
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                        StatItem("Params", "${uiState.paramsM}M")
                        StatItem("Memory", "${uiState.memoryUsageMB}MB")
                        StatItem("Uptime", uiState.uptime)
                    }
                }
            }

            Spacer(Modifier.height(16.dp))

            // Connection status
            Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(
                containerColor = if (uiState.isConnectedToServer) Color(0xFFE8F5E9) else Color(0xFFFFF3E0))) {
                Row(Modifier.fillMaxWidth().padding(16.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically) {
                    Column {
                        Text("Server", fontWeight = FontWeight.Bold)
                        Text(uiState.serverAddress, style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    Box(Modifier.size(12.dp).background(
                        if (uiState.isConnectedToServer) Color(0xFF4CAF50) else Color(0xFFFF9800),
                        CircleShape
                    ))
                }
            }

            // Training progress bar
            if (uiState.isTrainingActive && uiState.totalRounds > 0) {
                Spacer(Modifier.height(16.dp))
                LinearProgressIndicator(
                    progress = { uiState.currentRound.toFloat() / uiState.totalRounds },
                    modifier = Modifier.fillMaxWidth().height(8.dp).clip(RoundedCornerShape(4.dp)),
                )
            }
        }
    }
}

@Composable
fun StatItem(label: String, value: String) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(value, style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.primary)
        Text(label, style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}
