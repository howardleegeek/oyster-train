package ai.oyster.train.ui

import ai.oyster.train.ModelType
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
    val ui by viewModel.uiState.collectAsState()
    var serverInput by remember { mutableStateOf("10.0.2.2:8080") }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Oyster Train") },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                )
            )
        }
    ) { padding ->
        Column(
            Modifier.fillMaxSize().padding(padding).padding(16.dp)
                .verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            // ── Status indicator ──
            Row(verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Box(Modifier.size(16.dp).background(
                    if (ui.isTrainingActive) Color(0xFF4CAF50) else Color(0xFF9E9E9E), CircleShape))
                Text(
                    if (ui.isTrainingActive) "Training Active" else "Idle",
                    style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
            }

            Spacer(Modifier.height(12.dp))

            // ── Model selector chips ──
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                ModelType.entries.forEach { model ->
                    FilterChip(
                        selected = ui.selectedModel == model,
                        onClick = { viewModel.selectModel(model) },
                        label = { Text(when (model) {
                            ModelType.LEWM -> "LeWM"
                            ModelType.QWEN -> "Qwen 0.5B"
                            ModelType.BOTH -> "Both"
                        }) },
                        enabled = !ui.isTrainingActive,
                    )
                }
            }

            Spacer(Modifier.height(12.dp))

            // ── LeWM model card ──
            if (ui.selectedModel == ModelType.LEWM || ui.selectedModel == ModelType.BOTH) {
                ModelCard(
                    title = "LeWM JEPA — World Model",
                    subtitle = "Vision + action prediction | ${String.format("%.1f", ui.lewm.paramsM)}M params",
                    ready = ui.lewm.ready,
                    training = ui.lewm.training,
                    status = ui.lewm,
                    accentColor = Color(0xFF1565C0),
                )
                Spacer(Modifier.height(8.dp))
            }

            // ── Qwen model card ──
            if (ui.selectedModel == ModelType.QWEN || ui.selectedModel == ModelType.BOTH) {
                if (!ui.qwenDownloaded) {
                    // Download prompt
                    Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(
                        containerColor = Color(0xFFFFF3E0))) {
                        Column(Modifier.padding(16.dp)) {
                            Text("Qwen2.5-0.5B — Not Downloaded", fontWeight = FontWeight.Bold)
                            Text("~350MB GGUF model needed for on-device LLM",
                                style = MaterialTheme.typography.bodySmall)
                            Spacer(Modifier.height(8.dp))
                            if (ui.qwenDownloading) {
                                LinearProgressIndicator(Modifier.fillMaxWidth())
                                Text(ui.qwenDownloadProgress, style = MaterialTheme.typography.bodySmall)
                            } else {
                                Button(onClick = { viewModel.downloadQwen() },
                                    shape = RoundedCornerShape(8.dp)) {
                                    Text("Download Model")
                                }
                            }
                        }
                    }
                } else {
                    ModelCard(
                        title = "Qwen2.5-0.5B — Language Model",
                        subtitle = "LoRA rank-8 fine-tuning | ${String.format("%.1f", ui.qwen.tokensPerSec)} tok/s",
                        ready = ui.qwen.ready,
                        training = ui.qwen.training,
                        status = ui.qwen,
                        accentColor = Color(0xFF6A1B9A),
                    )
                }
                Spacer(Modifier.height(8.dp))
            }

            // ── Server + controls ──
            OutlinedTextField(
                value = serverInput,
                onValueChange = { serverInput = it },
                label = { Text("Flower Server (ip:port)") },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
                enabled = !ui.isTrainingActive,
            )

            Spacer(Modifier.height(12.dp))

            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                val canStart = !ui.isTrainingActive && when (ui.selectedModel) {
                    ModelType.LEWM -> ui.lewm.ready
                    ModelType.QWEN -> ui.qwenDownloaded && ui.qwen.ready
                    ModelType.BOTH -> ui.lewm.ready  // Qwen optional
                }
                Button(
                    onClick = { viewModel.startTraining(serverInput) },
                    enabled = canStart,
                    modifier = Modifier.weight(1f),
                    shape = RoundedCornerShape(8.dp),
                ) { Text("Join Network") }

                Button(
                    onClick = { viewModel.stopTraining() },
                    enabled = ui.isTrainingActive,
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error),
                    shape = RoundedCornerShape(8.dp),
                ) { Text("Stop") }
            }

            Spacer(Modifier.height(12.dp))

            // ── Uptime + error ──
            if (ui.isTrainingActive) {
                Text("Uptime: ${ui.uptime}", style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            if (ui.errorMessage.isNotEmpty()) {
                Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(
                    containerColor = Color(0xFFFFEBEE))) {
                    Text(ui.errorMessage, Modifier.padding(12.dp), color = Color(0xFFC62828),
                        style = MaterialTheme.typography.bodySmall)
                }
            }
        }
    }
}

@Composable
fun ModelCard(
    title: String,
    subtitle: String,
    ready: Boolean,
    training: Boolean,
    status: ModelStatus,
    accentColor: Color,
) {
    Card(
        Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = if (training) accentColor.copy(alpha = 0.08f)
            else MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Column(Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Box(Modifier.size(10.dp).background(
                    when {
                        training -> Color(0xFF4CAF50)
                        ready -> accentColor
                        else -> Color(0xFFFF9800)
                    }, CircleShape))
                Text(title, fontWeight = FontWeight.Bold,
                    style = MaterialTheme.typography.titleSmall)
            }
            Text(subtitle, style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)

            if (training || status.round > 0) {
                Spacer(Modifier.height(12.dp))
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                    StatItem("Round", "${status.round}/${status.totalRounds}")
                    StatItem("Loss", String.format("%.4f", status.loss))
                    StatItem("Step", "${status.step}/${status.totalSteps}")
                }
                if (training && status.totalRounds > 0) {
                    Spacer(Modifier.height(8.dp))
                    LinearProgressIndicator(
                        progress = { status.round.toFloat() / status.totalRounds },
                        modifier = Modifier.fillMaxWidth().height(4.dp).clip(RoundedCornerShape(2.dp)),
                        color = accentColor,
                    )
                }
            }

            if (status.error.isNotEmpty()) {
                Spacer(Modifier.height(4.dp))
                Text(status.error, color = Color(0xFFC62828),
                    style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}

@Composable
fun StatItem(label: String, value: String) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(value, style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.primary)
        Text(label, style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}
