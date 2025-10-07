"""
Bot Integration
Integrates auto-update system with the main Telegram bot
"""

import asyncio
import logging
from typing import Optional
from telegram import Update
from telegram.ext import ContextTypes, CommandHandler
from .scheduler import AutoUpdateScheduler

logger = logging.getLogger(__name__)

class AutoUpdateIntegration:
    """Integrates auto-update system with Telegram bot"""
    
    def __init__(self):
        self.scheduler: Optional[AutoUpdateScheduler] = None
        self.is_enabled = True  # Can be controlled by admin
        
    async def start_auto_updates(self, update_interval_hours: int = 24):
        """Start the auto-update scheduler"""
        if self.scheduler and self.scheduler.is_running:
            logger.warning("Auto-update scheduler is already running")
            return False
        
        logger.info(f"🚀 Starting auto-update system (interval: {update_interval_hours}h)...")
        
        self.scheduler = AutoUpdateScheduler(
            update_interval_hours=update_interval_hours,
            auto_apply=self.is_enabled,
            min_score_threshold=60.0
        )
        
        await self.scheduler.start()
        return True
    
    async def stop_auto_updates(self):
        """Stop the auto-update scheduler"""
        if not self.scheduler:
            return False
            
        await self.scheduler.stop()
        logger.info("⏹️ Auto-update system stopped")
        return True
    
    async def run_manual_update(self, dry_run: bool = False) -> dict:
        """Run a manual update cycle"""
        if not self.scheduler:
            self.scheduler = AutoUpdateScheduler(auto_apply=not dry_run)
        
        return await self.scheduler.run_manual_update(dry_run=dry_run)
    
    def get_status(self) -> dict:
        """Get auto-update system status"""
        if not self.scheduler:
            return {
                'running': False,
                'enabled': self.is_enabled,
                'message': 'Auto-update system not initialized'
            }
        
        status = self.scheduler.get_status()
        status['enabled'] = self.is_enabled
        return status


# Global instance
auto_update_integration = AutoUpdateIntegration()


# Admin command handlers
async def cmd_auto_update_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show auto-update system status"""
    status = auto_update_integration.get_status()
    
    message = "🔄 **Auto-Update System Status**\\n\\n"
    message += f"• Running: {'✅ Yes' if status.get('running') else '❌ No'}\\n"
    message += f"• Enabled: {'✅ Yes' if status.get('enabled') else '❌ No'}\\n"
    
    if status.get('update_interval_hours'):
        message += f"• Update Interval: {status['update_interval_hours']}h\\n"
    
    if status.get('last_update'):
        message += f"• Last Update: {status['last_update']}\\n"
    
    if status.get('next_update'):
        message += f"• Next Update: {status['next_update']}\\n"
    
    if status.get('min_score_threshold'):
        message += f"• Min Score Threshold: {status['min_score_threshold']}\\n"
    
    # Show recent updates
    if status.get('update_history'):
        message += "\\n📜 **Recent Updates:**\\n"
        for i, update_record in enumerate(status['update_history'][:3], 1):
            timestamp = update_record.get('timestamp', 'Unknown')
            success = update_record.get('success', False)
            status_icon = '✅' if success else '❌'
            message += f"  {i}\\. {status_icon} {timestamp[:19]}\\n"
    
    await update.message.reply_text(message, parse_mode='MarkdownV2')


async def cmd_run_manual_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Run manual model update"""
    await update.message.reply_text("🔍 Starting manual model update\\.\\.\\.")
    
    try:
        result = await auto_update_integration.run_manual_update(dry_run=False)
        
        message = "✅ **Manual Update Completed**\\n\\n"
        if result.get('last_update'):
            message += f"• Completed: {result['last_update']}\\n"
        
        await update.message.reply_text(message, parse_mode='MarkdownV2')
        
    except Exception as e:
        logger.error(f"Manual update failed: {e}")
        await update.message.reply_text(f"❌ Update failed: {str(e)}")


async def cmd_test_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Run dry-run update (no changes applied)"""
    await update.message.reply_text("🧪 Running test update \\(dry run\\)\\.\\.\\.")
    
    try:
        result = await auto_update_integration.run_manual_update(dry_run=True)
        
        message = "✅ **Test Update Completed**\\n\\n"
        message += "No changes were applied \\(dry run mode\\)\\n"
        
        await update.message.reply_text(message, parse_mode='MarkdownV2')
        
    except Exception as e:
        logger.error(f"Test update failed: {e}")
        await update.message.reply_text(f"❌ Test failed: {str(e)}")


def get_auto_update_handlers():
    """Get command handlers for auto-update system"""
    return [
        CommandHandler('autoupdate_status', cmd_auto_update_status),
        CommandHandler('manual_update', cmd_run_manual_update),
        CommandHandler('test_update', cmd_test_update)
    ]
