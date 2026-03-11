---
task_id: S07-data-pipeline
project: oyster-train
priority: 2
estimated_minutes: 40
depends_on: []
modifies: ["data/"]
executor: glm
---

## Goal
Build the training data pipeline: data collection strategy, tokenization, privacy-preserving preprocessing, and non-IID data management for 10K phone federated learning.

## Context
- 10K UBS1 phones with diverse user data (text input, app usage, etc.)
- Model: Qwen2.5-1.5B-Instruct (Chinese + English bilingual)
- Privacy: raw user data never leaves the device
- Non-IID: each phone has different data distribution
- Need server-side data for initial model pre-fine-tuning before federated rounds

## Deliverables

### data/tokenizer.py
- Wrapper around Qwen2.5 tokenizer (from transformers library)
- `OysterTokenizer` class:
  - `encode(text, max_length=256) -> List[int]`
  - `decode(tokens) -> str`
  - `create_training_pairs(text) -> List[InputOutputPair]`
    - For instruction tuning: split into instruction/response pairs
    - For language modeling: create sliding window chunks
  - `batch_encode(texts, batch_size=4) -> DataLoader`
- Handle both Chinese and English text
- Export tokenizer vocab for on-device use (SentencePiece model file)

### data/privacy.py
- Privacy-preserving data preprocessing
- `sanitize_text(text) -> str`:
  - Remove PII (emails, phone numbers, addresses) using regex patterns
  - Remove URLs and IP addresses
  - Replace names with [NAME] placeholder (simple NER)
- `differential_privacy_noise(gradients, epsilon=1.0, delta=1e-5) -> Tensor`:
  - Add calibrated Gaussian noise for (ε,δ)-differential privacy
  - Clip gradient norms before adding noise
  - Return noised gradients
- `SecureAggregation` class (placeholder for future implementation):
  - Document the protocol for secure aggregation
  - Stub methods: generate_mask(), apply_mask(), aggregate_masked()

### data/data_sources.py
- Server-side training data collection for initial fine-tuning:
  - `download_alpaca_cleaned() -> Dataset` (instruction tuning)
  - `download_wikitext() -> Dataset` (language modeling)
  - `download_chinese_alpaca() -> Dataset` (Chinese instruction tuning)
  - `create_mixed_dataset(datasets, weights) -> Dataset`
    - Mix datasets with specified weights
    - Shuffle and split into train/validation
- On-device data specification:
  - Document what data types phones can collect
  - Define data format for on-device storage
  - Schema: `{"text": str, "source": str, "timestamp": int, "language": str}`

### data/non_iid.py
- Tools for creating and analyzing non-IID data distributions
- `DirichletPartitioner(num_clients, alpha=0.5)`:
  - Partition dataset into non-IID shards using Dirichlet distribution
  - alpha < 1.0 = more heterogeneous (realistic)
  - alpha > 1.0 = more homogeneous (easier to train)
  - Visualize distribution with matplotlib
- `analyze_heterogeneity(client_data_list) -> HeterogeneityMetrics`:
  - Earth Mover's Distance between client distributions
  - Label distribution skew per client
  - Vocabulary overlap between clients
- `FedProxRegularizer(mu=0.01)`:
  - Implements proximal term to handle non-IID (FedProx algorithm)
  - `compute_proximal_loss(local_model, global_model) -> Tensor`

### data/requirements.txt
- transformers>=4.40
- datasets
- torch
- numpy
- matplotlib
- sentencepiece

## Constraints
- Python 3.10+
- Must handle both Chinese and English text
- PII removal must be robust (regex patterns for common formats)
- Differential privacy implementation must be mathematically correct
- All functions must have docstrings and type hints
- Write tests in tests/test_data.py

## Acceptance Criteria
- [ ] Tokenizer correctly encodes/decodes Chinese and English text
- [ ] PII sanitization removes emails, phones, URLs
- [ ] Differential privacy noise is correctly calibrated
- [ ] Non-IID partitioning creates skewed distributions (verify visually)
- [ ] FedProx regularizer computes correct proximal loss
- [ ] All tests pass
