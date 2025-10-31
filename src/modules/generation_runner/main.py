#!/usr/bin/env python3
""" Runner module for orchestrating pipeline step execution """

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

from pipeline_utilities.args import BaseArgumentParser, parse_arguments
from pipeline_utilities.configuration import load_configuration
from pipeline_utilities.logs import EnhancedLogger
from pipeline_utilities.paths import Project


class RunnerArgumentParser(BaseArgumentParser):
    """ Custom argument parser for runner module """

    def add_base_arguments(self) -> None:
        """ Override to customize --step argument """
        self.parser.add_argument(
            '--log-file',
            type=str,
            default='pipeline.log',
            help='Log file path (default: pipeline.log)'
        )
        self.parser.add_argument(
            '--step',
            type=str,
            default='all',
            help='Steps to execute: "all" or comma-separated numbers (e.g., "1,2,4")'
        )


def parse_step_selection(step_arg: str, total_steps: int) -> List[int]:
    """ Parse step argument and return list of step indices """
    if step_arg.lower() == 'all':
        return list(range(total_steps))

    # Parse comma-separated numbers
    try:
        step_numbers = [int(x.strip()) for x in step_arg.split(',')]
    except ValueError as exc:
        raise ValueError(f"Invalid step numbers '{step_arg}': must be comma-separated integers") from exc

    # Validate numbers are in range [1, total_steps]
    for step_num in step_numbers:
        if step_num < 1 or step_num > total_steps:
            raise ValueError(f"Step number {step_num} is out of range (1-{total_steps})")

    # Convert to 0-based indices
    return [num - 1 for num in step_numbers]


def resolve_steps(config_data, step_indices: List[int]) -> List[Tuple[str, str]]:
    """ Resolve step indices to (step_name, module_name) tuples """
    # Get ordered list of steps from config (preserving JSON order)
    step_names = list(config_data.steps.keys())

    result = []
    for idx in step_indices:
        if idx >= len(step_names):
            raise ValueError(f"Step index {idx} is out of range (0-{len(step_names)-1})")

        step_name = step_names[idx]
        step_config = config_data.steps[step_name]
        module_name = step_config.module

        result.append((step_name, module_name))

    return result


def execute_module(module_name: str, step_name: str, log_file_path: Path, logger: EnhancedLogger) -> None:
    """ Execute a module's main.py with its own venv """
    project_root = Project.get_root()

    # Construct paths
    module_dir = project_root / "src" / "modules" / module_name
    main_py = module_dir / "main.py"
    venv_python = module_dir / "venv" / "bin" / "python"

    # Validate paths exist
    if not module_dir.exists():
        raise FileNotFoundError(f"Module directory not found: {module_dir}")

    if not main_py.exists():
        raise FileNotFoundError(f"Module main.py not found: {main_py}")

    if not venv_python.exists():
        raise FileNotFoundError(f"Module venv python not found: {venv_python}")

    # Build command
    cmd = [
        str(venv_python),
        str(main_py),
        '--step', step_name,
        '--log-file', str(log_file_path)
    ]

    logger.info(f"Executing: {' '.join(cmd)}")

    try:
        # Execute subprocess with streaming output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            cwd=str(module_dir)  # Run from module directory
        )

        # Stream output line-by-line to console
        for line in process.stdout:
            # Print without newline since line already contains it
            print(line, end='')
            sys.stdout.flush()  # Ensure immediate output

        # Wait for process to complete
        return_code = process.wait()

        # Check return code
        if return_code != 0:
            raise RuntimeError(f"Module {module_name} failed with exit code {return_code}")

        logger.debug(f"Module {module_name} completed successfully")

    except subprocess.SubprocessError as exc:
        raise RuntimeError(f"Failed to execute module {module_name}: {exc}") from exc


def main() -> None:
    """ Main entry point """
    # Parse arguments
    args = parse_arguments(
        "Pipeline runner for orchestrating step execution",
        RunnerArgumentParser
    )

    try:
        # Load configuration first to get debug setting
        config_path = Project.get_configuration()
        config_loader = load_configuration(config_path)
        config_data = config_loader.data

        # Setup logging with debug setting from config
        logger = EnhancedLogger.setup_pipeline_logging(
            log_file=args.log_file,
            debug=config_data.debug or False,
            script_name="runner"
        )

        logger.info(f"Loaded configuration: {config_data.name}")

        # Convert log file to absolute path
        log_file_path = Path(args.log_file).resolve()
        logger.info(f"Using log file: {log_file_path}")

        # Delete existing log file to start fresh
        if log_file_path.exists():
            log_file_path.unlink()
            logger.info("Deleted existing log file")

        # Get total number of steps
        total_steps = len(config_data.steps)
        logger.info(f"Total steps available: {total_steps}")

        # Parse step selection
        step_indices = parse_step_selection(args.step, total_steps)
        logger.info(f"Selected step indices: {[i+1 for i in step_indices]}")  # Show 1-based for user

        # Resolve steps to (step_name, module_name) tuples
        steps_to_execute = resolve_steps(config_data, step_indices)

        logger.info("-" * 60)
        logger.info("- STARTING PIPELINE EXECUTION")
        logger.info("-" * 60)
        logger.info(f"Executing {len(steps_to_execute)} steps")

        # Execute each step
        for i, (step_name, module_name) in enumerate(steps_to_execute, 1):
            logger.info(f"Module: {module_name}")

            try:
                execute_module(module_name, step_name, log_file_path, logger)
            except Exception as exc:
                logger.error(f"Step {step_name} failed: {exc}")
                logger.header("Pipeline execution failed")
                raise

        logger.header("Pipeline execution completed successfully")
        logger.info(f"Executed {len(steps_to_execute)} steps")

    except Exception as exc:
        logger.error(f"Pipeline runner failed: {exc}")
        raise


if __name__ == "__main__":
    main()
