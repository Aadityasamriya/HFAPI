# Auto-Update System

Automatically discovers, evaluates, and implements the best free AI models every 24 hours.

## 🎯 Overview

The auto-update system continuously researches and implements the best performing free AI models from Hugging Face Hub, ensuring your bot always uses the most accurate and capable models available.

### Key Features

- **Automatic Model Discovery**: Searches 100+ models across 9 task categories using Hugging Face Hub API
- **Intelligent Scoring**: Multi-criteria evaluation (popularity 30%, performance 35%, recency 15%, compatibility 20%)
- **Safe Updates**: Automatic configuration backup before changes
- **24-Hour Scheduling**: Runs updates automatically every 24 hours
- **Admin Control**: Telegram commands for manual updates and monitoring
- **Dry-Run Mode**: Test updates without applying changes

## 🏗️ Architecture

### Components

1. **ModelDiscoveryEngine** (`model_discovery.py`)
   - Discovers models using HF Hub API
   - Filters for free-tier compatible models
   - Supports 9 task categories (text generation, code generation, Q&A, etc.)

2. **PerformanceEvaluator** (`performance_evaluator.py`)
   - Scores models based on multiple criteria
   - Generates evaluation reports
   - Provides recommendations

3. **ConfigUpdater** (`config_updater.py`)
   - Updates bot configuration automatically
   - Maintains update history (last 30 updates)
   - Supports rollback to previous configuration

4. **AutoUpdateScheduler** (`scheduler.py`)
   - Runs updates every 24 hours
   - Handles manual updates
   - Comprehensive logging

5. **Bot Integration** (`bot_integration.py`)
   - Telegram admin commands
   - Status monitoring
   - Manual control

## 🚀 Usage

### Running the Auto-Update System

#### Standalone Mode
```bash
# Auto mode (24-hour scheduler)
python -m bot.auto_update.runner --mode auto

# Manual update (run once)
python -m bot.auto_update.runner --mode manual

# Dry-run (test without applying changes)
python -m bot.auto_update.runner --mode dry-run
```

#### Integrated with Bot
```python
from bot.auto_update.bot_integration import auto_update_integration

# Start auto-updates
await auto_update_integration.start_auto_updates(update_interval_hours=24)

# Run manual update
result = await auto_update_integration.run_manual_update(dry_run=False)

# Get status
status = auto_update_integration.get_status()
```

### Admin Commands (Telegram)

- `/autoupdate_status` - View system status and update history
- `/manual_update` - Run manual model update immediately
- `/test_update` - Run dry-run update (no changes applied)

## 📊 Scoring Criteria

Models are scored on 4 criteria (0-100 scale):

1. **Popularity (30%)**
   - Downloads (log scale)
   - Likes/Stars

2. **Performance (35%)**
   - Benchmark metrics (HumanEval, MMLU, etc.)
   - Task-specific performance

3. **Recency (15%)**
   - Last modified date
   - Decay function: 100 (<30d) → 20 (>365d)

4. **Compatibility (20%)**
   - Free-tier license
   - Inference endpoint support

**Minimum Score Threshold**: 60.0 (configurable)

## 🔍 Model Discovery

### Supported Tasks
- `text-generation` - General text generation
- `code-generation` - Code generation and completion
- `conversational` - Chat and dialogue
- `question-answering` - Q&A systems
- `summarization` - Text summarization
- `translation` - Language translation
- `text2text-generation` - Text transformation
- `fill-mask` - Text completion
- `feature-extraction` - Embeddings

### Preferred Model Families
- Qwen (Qwen2.5, Qwen2.5-Coder)
- DeepSeek (DeepSeek-Coder-V2)
- Mistral (Mistral-7B)
- Meta Llama (Llama 3.3)
- Phi, Gemma, Falcon, StarCoder

## 📝 Configuration

### Scheduler Settings
```python
scheduler = AutoUpdateScheduler(
    update_interval_hours=24,  # Update frequency
    auto_apply=True,           # Auto-apply updates
    min_score_threshold=60.0   # Minimum model score
)
```

### Task-to-Intent Mapping
```python
TASK_TO_INTENT_MAP = {
    'text-generation': ['text_generation', 'conversation', 'creative_writing'],
    'code-generation': ['code_generation'],
    'conversational': ['conversation'],
    'question-answering': ['question_answering'],
    # ... more mappings
}
```

## 🛡️ Safety Features

- **Configuration Backup**: Automatic backup before updates
- **Rollback Support**: Restore previous configuration
- **Update History**: Track last 30 updates
- **Dry-Run Mode**: Test updates without changes
- **Error Handling**: Comprehensive error recovery
- **Non-Blocking**: Async operations with thread offloading

## 📈 Update Process

1. **Discovery Phase**
   - Query Hugging Face Hub API
   - Filter by task, license, popularity
   - Extract model metadata

2. **Evaluation Phase**
   - Score each model (0-100)
   - Apply scoring criteria
   - Filter by threshold

3. **Configuration Phase**
   - Backup current config
   - Generate new mappings
   - Update config files
   - Log changes

4. **Verification**
   - Validate updates
   - Log results
   - Update history

## 📊 Monitoring

### Status Information
```python
status = auto_update_integration.get_status()

# Returns:
{
    'running': True/False,
    'enabled': True/False,
    'update_interval_hours': 24,
    'last_update': '2025-10-07T...',
    'next_update': '2025-10-08T...',
    'min_score_threshold': 60.0,
    'update_history': [...]
}
```

### Update History
Stored in: `bot/auto_update/update_history.json`

Each entry contains:
- Timestamp
- Success status
- Models discovered/qualified/updated
- Changes made
- Errors (if any)

## 🔧 Troubleshooting

### No Models Discovered
- Check Hugging Face API connectivity
- Verify HF_TOKEN environment variable
- Check task category names

### Low Quality Scores
- Adjust `min_score_threshold` (default: 60.0)
- Review scoring weights
- Check model metadata availability

### Configuration Not Updated
- Check backup file exists
- Verify write permissions
- Review error logs in update history

### Event Loop Blocking
- Ensure asyncio.to_thread() is used for HF API calls
- Check for synchronous operations in async code

## 🎯 Best Practices

1. **Start with Dry-Run**: Test updates before applying
2. **Monitor Logs**: Check update history regularly
3. **Backup Configuration**: Keep manual backups
4. **Adjust Thresholds**: Tune scores based on needs
5. **Review Updates**: Check new models before deployment

## 📚 API Reference

### ModelDiscoveryEngine
```python
engine = ModelDiscoveryEngine()
candidates = await engine.discover_models(task='text-generation', limit=50)
summary = engine.get_discovery_summary()
```

### PerformanceEvaluator
```python
evaluator = PerformanceEvaluator()
scores = evaluator.evaluate_models(candidates)
report = evaluator.generate_evaluation_report(scores)
```

### ConfigUpdater
```python
updater = ConfigUpdater()
result = updater.update_model_configuration(scores, dry_run=False)
history = updater.get_update_history(limit=10)
updater.rollback_to_backup()
```

### AutoUpdateScheduler
```python
scheduler = AutoUpdateScheduler(update_interval_hours=24)
await scheduler.start()
result = await scheduler.run_manual_update(dry_run=False)
status = scheduler.get_status()
await scheduler.stop()
```

## 📄 License

Part of the Hugging Face By AadityaLabs AI Telegram Bot project.
