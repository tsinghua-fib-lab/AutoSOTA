"""
Metric Runner: Orchestrates running gen_metric.py with different parameter combinations from YAML config.
Supports GPU parallelization with proper queue management and job prioritization.
"""

import click


from semantic_isotropy.pipeline.utils import init_logger
from semantic_isotropy.pipeline.jobs import MetricRunner

logger = init_logger(__name__, 'INFO')


@click.command()
@click.option('--config', '-c', required=True, type=click.Path(exists=True), help='Path to YAML configuration file')
@click.option('--dryrun', is_flag=True, help='Show commands that would be run without executing')
@click.option('-q', '--quiet', is_flag=True, help='In dry run mode, show only commands without additional logging')
def main(config: str, dryrun: bool, quiet: bool):
    """Run metric generation with multiple parameter combinations"""
    
    try:
        runner = MetricRunner(config)
        
        # Run with dryrun parameter (this handles both CLI and config-level dryrun)
        results = runner.run_all(dryrun=dryrun, quiet=quiet)
        
        # Results summary is now handled by run_all method
        # No additional summary needed here
    
    except Exception as e:
        logger.error(f"Failed to run metric runner: {e}")
        raise click.ClickException(str(e))

if __name__ == "__main__":
    main()
