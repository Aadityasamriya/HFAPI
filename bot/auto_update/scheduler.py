"""
Auto-Update Scheduler
Runs model discovery and updates every 24 hours
"""

import asyncio
import logging
from typing import Optional
from datetime import datetime, timedelta
from .model_discovery import ModelDiscoveryEngine
from .performance_evaluator import PerformanceEvaluator
from .config_updater import ConfigUpdater

logger = logging.getLogger(__name__)


class AutoUpdateScheduler:
    """Schedules and manages automatic model updates"""
    
    def __init__(
        self, 
        update_interval_hours: int = 24,
        auto_apply: bool = True,
        min_score_threshold: float = 60.0
    ):
        """
        Initialize scheduler
        
        Args:
            update_interval_hours: Hours between updates (default: 24)
            auto_apply: Automatically apply updates (default: True)
            min_score_threshold: Minimum score to consider a model (default: 60.0)
        """
        self.update_interval = timedelta(hours=update_interval_hours)
        self.auto_apply = auto_apply
        self.min_score_threshold = min_score_threshold
        
        self.discovery_engine = ModelDiscoveryEngine()
        self.evaluator = PerformanceEvaluator()
        self.config_updater = ConfigUpdater()
        
        self.is_running = False
        self.last_update: Optional[datetime] = None
        self.next_update: Optional[datetime] = None
        self._task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the auto-update scheduler"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        logger.info(f"🚀 Auto-update scheduler started (interval: {self.update_interval.total_seconds()/3600}h)")
        
        # Run immediately on start
        await self._run_update_cycle()
        
        # Schedule periodic updates
        self._task = asyncio.create_task(self._schedule_loop())
    
    async def stop(self):
        """Stop the scheduler"""
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("⏹️ Auto-update scheduler stopped")
    
    async def _schedule_loop(self):
        """Main scheduling loop"""
        while self.is_running:
            try:
                # Calculate next update time
                self.next_update = datetime.now() + self.update_interval
                
                # Wait until next update
                logger.info(f"⏰ Next update scheduled for: {self.next_update.strftime('%Y-%m-%d %H:%M:%S')}")
                await asyncio.sleep(self.update_interval.total_seconds())
                
                # Run update cycle
                if self.is_running:
                    await self._run_update_cycle()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in scheduler loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _run_update_cycle(self):
        """Execute complete update cycle"""
        cycle_start = datetime.now()
        logger.info("=" * 60)
        logger.info(f"🔄 Starting auto-update cycle at {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
        update_result = {
            'success': False,
            'timestamp': cycle_start.isoformat(),
            'models_discovered': 0,
            'models_qualified': 0,
            'models_updated': {},
            'errors': []
        }
        
        try:
            # Step 1: Discover models
            logger.info("📡 Step 1: Discovering models...")
            candidates = await self.discovery_engine.discover_models(limit=100)
            update_result['models_discovered'] = len(candidates)
            logger.info(f"✅ Discovered {len(candidates)} model candidates")
            
            if not candidates:
                logger.warning("⚠️ No models discovered, skipping update")
                update_result['errors'].append("No models discovered")
                return update_result
            
            # Step 2: Evaluate models
            logger.info("📊 Step 2: Evaluating model performance...")
            scores = self.evaluator.evaluate_models(candidates)
            
            # Filter by minimum score
            qualified_scores = [s for s in scores if s.total_score >= self.min_score_threshold]
            update_result['models_qualified'] = len(qualified_scores)
            logger.info(f"✅ {len(qualified_scores)}/{len(scores)} models meet quality threshold ({self.min_score_threshold})")
            
            if not qualified_scores:
                logger.warning("⚠️ No models meet minimum score threshold, skipping update")
                update_result['errors'].append("No models meet minimum score threshold")
                return update_result
            
            # Step 3: Generate evaluation report
            report = self.evaluator.generate_evaluation_report(qualified_scores)
            self._log_evaluation_report(report)
            update_result['evaluation_report'] = report
            
            # Step 4: Update configuration
            if self.auto_apply:
                logger.info("⚙️ Step 3: Updating configuration...")
                config_result = self.config_updater.update_model_configuration(
                    qualified_scores,
                    dry_run=False
                )
                
                if config_result['success']:
                    logger.info(f"✅ Configuration updated with {len(config_result['models_updated'])} models")
                    self._log_update_summary(config_result)
                    update_result['success'] = True
                    update_result['models_updated'] = config_result['models_updated']
                    update_result['changes'] = config_result.get('changes', [])
                else:
                    logger.error(f"❌ Configuration update failed: {config_result.get('errors', [])}")
                    update_result['errors'].extend(config_result.get('errors', []))
            else:
                logger.info("🔍 Dry run mode - configuration not updated")
                config_result = self.config_updater.update_model_configuration(
                    qualified_scores,
                    dry_run=True
                )
                self._log_update_summary(config_result)
                update_result['success'] = True
                update_result['models_updated'] = config_result['models_updated']
                update_result['changes'] = config_result.get('changes', [])
            
            # Update timestamp
            self.last_update = datetime.now()
            
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            update_result['duration_seconds'] = cycle_duration
            logger.info("=" * 60)
            logger.info(f"✅ Update cycle completed in {cycle_duration:.1f}s")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"❌ Update cycle failed: {e}", exc_info=True)
            update_result['errors'].append(str(e))
        
        return update_result
    
    async def run_manual_update(self, dry_run: bool = False) -> dict:
        """
        Run update cycle manually
        
        Args:
            dry_run: If True, don't apply changes
            
        Returns:
            Dictionary with update results
        """
        logger.info(f"🔧 Running manual update (dry_run={dry_run})...")
        
        prev_auto_apply = self.auto_apply
        self.auto_apply = not dry_run
        
        result = await self._run_update_cycle()
        
        self.auto_apply = prev_auto_apply
        
        # Add scheduler status to result
        if result:
            result['last_update'] = self.last_update.isoformat() if self.last_update else None
            result['next_update'] = self.next_update.isoformat() if self.next_update else None
            result['auto_apply'] = self.auto_apply
        
        return result or {
            'success': False,
            'errors': ['Update cycle returned no result']
        }
    
    def _log_evaluation_report(self, report: dict):
        """Log evaluation report"""
        logger.info("\n📈 EVALUATION REPORT")
        logger.info(f"Total models evaluated: {report.get('total_models_evaluated', 0)}")
        
        logger.info("\n🏆 Overall Top 5 Models:")
        for i, model in enumerate(report.get('overall_top_5', [])[:5], 1):
            logger.info(f"  {i}. {model['model']} ({model['task']}) - Score: {model['score']}")
        
        logger.info("\n💡 Recommendations:")
        for rec in report.get('recommendations', []):
            logger.info(f"  {rec}")
    
    def _log_update_summary(self, update_result: dict):
        """Log update summary"""
        logger.info("\n📝 UPDATE SUMMARY")
        logger.info(f"Status: {'✅ Success' if update_result['success'] else '❌ Failed'}")
        logger.info(f"Models updated: {len(update_result.get('models_updated', {}))}")
        
        if update_result.get('models_updated'):
            logger.info("\nIntent → Model Mappings:")
            for intent, model in update_result['models_updated'].items():
                logger.info(f"  • {intent}: {model}")
        
        if update_result.get('changes'):
            logger.info("\nChanges made:")
            for change in update_result['changes']:
                logger.info(f"  • {change}")
        
        if update_result.get('errors'):
            logger.info("\nErrors:")
            for error in update_result['errors']:
                logger.error(f"  ❌ {error}")
    
    def get_status(self) -> dict:
        """Get scheduler status"""
        return {
            'is_running': self.is_running,
            'update_interval_hours': self.update_interval.total_seconds() / 3600,
            'auto_apply': self.auto_apply,
            'min_score_threshold': self.min_score_threshold,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'next_update': self.next_update.isoformat() if self.next_update else None,
            'update_history': self.config_updater.get_update_history(limit=5)
        }
