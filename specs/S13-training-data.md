---
task_id: S13-training-data
project: oyster-train
priority: 2
estimated_minutes: 30
depends_on: [S07]
modifies: ["data/dataset_prep.py", "data/download.py"]
executor: glm
---
## Goal
Build a training data preparation pipeline that downloads, processes, and shards instruction-following datasets for bilingual (Chinese + English) fine-tuning on 10K phones.

## Constraints
- Datasets (all open-source, HuggingFace):
  - English: `tatsu-lab/alpaca` (52K instructions)
  - Chinese: `silk-road/alpaca-data-gpt4-chinese` (52K instructions)
  - Mixed: `BAAI/COIG-CQIA` (subset, ~10K)
- Format: Alpaca-style `{"instruction": "", "input": "", "output": ""}`
- Non-IID sharding: Dirichlet distribution (alpha=0.5) across 10K clients
  - Each phone gets ~10-50 examples
  - Language distribution varies by phone (some mostly Chinese, some mostly English)
- Output: Pre-sharded .jsonl files, one per client batch (100 clients per file = 100 files)
- Must work with existing `data/data_loader.py` PhoneDataset class

## Deliverables
- `data/download.py` - Download and cache datasets from HuggingFace
- `data/dataset_prep.py` - Process, merge, and shard datasets:
  - Normalize all datasets to Alpaca format
  - Apply quality filters (min 10 chars instruction, max 2048 chars total)
  - Dirichlet non-IID sharding
  - Export as .jsonl shards
- `data/config.py` - Dataset configuration (paths, shard sizes, alpha)
- `tests/test_data_pipeline.py` - Test with small subset:
  - Download works (mock or tiny dataset)
  - Sharding produces correct number of shards
  - Each shard has valid format
  - Non-IID distribution is verified (KL divergence from uniform)

## Do NOT
- Actually download full datasets in tests (use mocks)
- Modify existing data/data_loader.py
- Store data in git (add data/*.jsonl to .gitignore)
