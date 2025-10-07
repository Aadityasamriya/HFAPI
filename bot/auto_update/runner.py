"""
Auto-Update Runner
Main entry point for the auto-update system
"""

import asyncio
import logging
import sys
from .scheduler import AutoUpdateScheduler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def run_auto_update_system():
    """Run the auto-update system"""
    logger.info("🚀 Initializing Auto-Update System...")
    
    # Create scheduler with 24-hour interval
    scheduler = AutoUpdateScheduler(
        update_interval_hours=24,
        auto_apply=True,  # Automatically apply updates
        min_score_threshold=60.0  # Only use models with score >= 60
    )
    
    try:
        # Start scheduler
        await scheduler.start()
        
        # Keep running
        logger.info("✅ Auto-update system is running. Press Ctrl+C to stop.")
        
        # Run indefinitely
        while True:
            await asyncio.sleep(3600)  # Check every hour
            
    except KeyboardInterrupt:
        logger.info("\n👋 Shutting down auto-update system...")
        await scheduler.stop()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        await scheduler.stop()
        sys.exit(1)


async def run_manual_update(dry_run: bool = False):
    """Run a manual update cycle"""
    logger.info("🔧 Running manual update cycle...")
    
    scheduler = AutoUpdateScheduler(auto_apply=not dry_run)
    result = await scheduler.run_manual_update(dry_run=dry_run)
    
    logger.info(f"\n✅ Manual update completed: {result}")
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-Update System Runner")
    parser.add_argument(
        '--mode',
        choices=['auto', 'manual', 'dry-run'],
        default='auto',
        help='Run mode: auto (24h scheduler), manual (run once), dry-run (simulate)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'auto':
        asyncio.run(run_auto_update_system())
    elif args.mode == 'manual':
        asyncio.run(run_manual_update(dry_run=False))
    elif args.mode == 'dry-run':
        asyncio.run(run_manual_update(dry_run=True))
