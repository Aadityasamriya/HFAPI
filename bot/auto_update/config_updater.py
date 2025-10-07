"""
Configuration Updater
Automatically updates bot configuration with best discovered models
"""

import json
import logging
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime
from .performance_evaluator import ModelScore

logger = logging.getLogger(__name__)


class ConfigUpdater:
    """Updates bot configuration with optimal models"""
    
    # Map tasks to intent types
    TASK_TO_INTENT_MAP = {
        'text-generation': ['text_generation', 'conversation', 'creative_writing'],
        'code-generation': ['code_generation'],
        'conversational': ['conversation'],
        'question-answering': ['question_answering'],
        'summarization': ['summarization'],
        'translation': ['translation'],
        'text2text-generation': ['text_generation']
    }
    
    def __init__(self, config_path: str = "bot/config.py"):
        self.config_path = Path(config_path)
        self.backup_path = Path("bot/config_backup.py")
        self.update_log_path = Path("bot/auto_update/update_history.json")
        self.update_history = self._load_update_history()
        
    def update_model_configuration(
        self, 
        scores: List[ModelScore],
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Update bot configuration with top models
        
        Args:
            scores: List of ModelScore objects
            dry_run: If True, only simulate updates without writing
            
        Returns:
            Dictionary with update results
        """
        update_result = {
            'success': False,
            'timestamp': datetime.now().isoformat(),
            'models_updated': {},
            'changes': [],
            'errors': []
        }
        
        try:
            # Group scores by task
            models_by_task = self._group_by_task(scores)
            
            # Generate new model mappings
            new_mappings = self._generate_model_mappings(models_by_task)
            
            if not new_mappings:
                update_result['errors'].append("No valid model mappings generated")
                return update_result
            
            # Backup current configuration
            if not dry_run:
                self._backup_config()
            
            # Update configuration
            changes = self._update_config_file(new_mappings, dry_run)
            
            update_result['success'] = True
            update_result['models_updated'] = new_mappings
            update_result['changes'] = changes
            
            # Log update
            if not dry_run:
                self._log_update(update_result)
            
            logger.info(f"✅ Configuration update {'simulated' if dry_run else 'completed'}: {len(new_mappings)} models")
            
        except Exception as e:
            error_msg = f"Configuration update failed: {e}"
            logger.error(error_msg)
            update_result['errors'].append(error_msg)
        
        return update_result
    
    def _group_by_task(self, scores: List[ModelScore]) -> Dict[str, List[ModelScore]]:
        """Group model scores by task"""
        grouped = {}
        for score in scores:
            if score.task not in grouped:
                grouped[score.task] = []
            grouped[score.task].append(score)
        
        # Sort each group by score
        for task in grouped:
            grouped[task].sort(key=lambda s: s.total_score, reverse=True)
        
        return grouped
    
    def _generate_model_mappings(self, models_by_task: Dict[str, List[ModelScore]]) -> Dict[str, str]:
        """Generate intent-to-model mappings"""
        mappings = {}
        
        for task, scores in models_by_task.items():
            if not scores:
                continue
            
            # Get top model for this task
            top_model = scores[0]
            
            # Map task to intent types
            intent_types = self.TASK_TO_INTENT_MAP.get(task, [])
            
            for intent in intent_types:
                # Use highest scoring model for each intent
                if intent not in mappings or top_model.total_score > 75:
                    mappings[intent] = top_model.model_id
        
        return mappings
    
    def _backup_config(self) -> bool:
        """Backup current configuration"""
        try:
            if self.config_path.exists():
                import shutil
                shutil.copy(self.config_path, self.backup_path)
                logger.info(f"✅ Configuration backed up to {self.backup_path}")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to backup config: {e}")
            return False
    
    def _update_config_file(self, new_mappings: Dict[str, str], dry_run: bool = False) -> List[str]:
        """Update config file with new model mappings"""
        changes = []
        
        try:
            # Read current config
            with open(self.config_path, 'r') as f:
                config_content = f.read()
            
            # Generate new model list
            new_models = self._generate_model_list(new_mappings)
            changes.append(f"Updated {len(new_models)} models in AVAILABLE_MODELS")
            
            # Update AVAILABLE_MODELS section
            updated_content = self._replace_model_list(config_content, new_models)
            
            # Update intent mappings if present
            updated_content = self._update_intent_mappings(updated_content, new_mappings)
            changes.append("Updated INTENT_MODEL_MAP with new mappings")
            
            # Write back if not dry run
            if not dry_run:
                with open(self.config_path, 'w') as f:
                    f.write(updated_content)
                logger.info(f"✅ Configuration file updated at {self.config_path}")
            else:
                logger.info(f"🔍 DRY RUN: Would update {self.config_path}")
            
        except Exception as e:
            error_msg = f"Failed to update config file: {e}"
            logger.error(error_msg)
            changes.append(f"ERROR: {error_msg}")
        
        return changes
    
    def _generate_model_list(self, mappings: Dict[str, str]) -> List[str]:
        """Generate unique list of models from mappings"""
        unique_models = list(set(mappings.values()))
        unique_models.sort()  # Alphabetical order
        return unique_models
    
    def _replace_model_list(self, content: str, new_models: List[str]) -> str:
        """Replace AVAILABLE_MODELS in config"""
        import re
        
        # Find AVAILABLE_MODELS section
        pattern = r'AVAILABLE_MODELS\s*=\s*\[(.*?)\]'
        
        # Create new models string
        model_entries = [f'        "{model}"' for model in new_models]
        new_models_str = "AVAILABLE_MODELS = [\n" + ",\n".join(model_entries) + "\n    ]"
        
        # Replace
        updated = re.sub(
            pattern,
            new_models_str,
            content,
            flags=re.DOTALL
        )
        
        return updated
    
    def _update_intent_mappings(self, content: str, mappings: Dict[str, str]) -> str:
        """Update INTENT_MODEL_MAP in config"""
        import re
        
        # Find INTENT_MODEL_MAP section
        pattern = r'INTENT_MODEL_MAP\s*=\s*\{(.*?)\}'
        
        # Create new mapping entries
        mapping_entries = []
        for intent, model in sorted(mappings.items()):
            mapping_entries.append(f'        "{intent}": "{model}"')
        
        new_map_str = "INTENT_MODEL_MAP = {\n" + ",\n".join(mapping_entries) + "\n    }"
        
        # Replace
        updated = re.sub(
            pattern,
            new_map_str,
            content,
            flags=re.DOTALL
        )
        
        return updated
    
    def _load_update_history(self) -> List[Dict[str, Any]]:
        """Load update history from JSON"""
        if self.update_log_path.exists():
            try:
                with open(self.update_log_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load update history: {e}")
        return []
    
    def _log_update(self, update_result: Dict[str, Any]):
        """Log update to history file"""
        try:
            self.update_history.append(update_result)
            
            # Keep only last 30 updates
            self.update_history = self.update_history[-30:]
            
            # Ensure directory exists
            self.update_log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to file
            with open(self.update_log_path, 'w') as f:
                json.dump(self.update_history, f, indent=2)
                
            logger.info(f"✅ Update logged to {self.update_log_path}")
            
        except Exception as e:
            logger.error(f"Failed to log update: {e}")
    
    def get_update_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent update history"""
        return self.update_history[-limit:]
    
    def rollback_to_backup(self) -> bool:
        """Rollback configuration to last backup"""
        try:
            if self.backup_path.exists():
                import shutil
                shutil.copy(self.backup_path, self.config_path)
                logger.info(f"✅ Configuration rolled back from {self.backup_path}")
                return True
            else:
                logger.error("No backup file found")
                return False
        except Exception as e:
            logger.error(f"❌ Rollback failed: {e}")
            return False
