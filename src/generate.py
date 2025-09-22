#!/usr/bin/env python3
"""
Main pipeline script for ambient track generation

This script orchestrates the execution of individual generation stages
in sequence, managing virtual environments and passing configuration files
"""

import re
import sys
import json
import shutil
import subprocess

from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from pipeline_utilities.logging_utils import setup_pipeline_logging, EnhancedLogger
from pipeline_utilities.generation import load_generation_config, Generation

from pipeline_utilities.paths import Paths


class _PipelineConfigurationError(Exception):
    """ Custom exception for pipeline configuration errors. """


def _prepare_workspace(log_path: Path, result_path: Path):
    """ Ensures a clean workspace by deleting the old log file and result directory """
    try:
        # Delete the old log file if it exists
        if log_path.exists():
            print("Removing old log file: %s", log_path)
            log_path.unlink()

        # Delete the old result directory if it exists
        if result_path.exists() and result_path.is_dir():
            print("Removing old result directory: %s", result_path)
            shutil.rmtree(result_path)
    except OSError as e:
        print(f"Error preparing workspace: {e}", file=sys.stderr)
        sys.exit(1)


def _setup_logging(pipeline_log_path: Path, debug: bool = False) -> EnhancedLogger:
    """ Set up logging configuration for the pipeline """
    return setup_pipeline_logging(
        log_file=str(pipeline_log_path),
        debug=debug,
        script_name="generate_pipeline"
    )


def _load_and_validate_config(config_path: Path) -> Generation:
    """ Load and validate the generation configuration file """
    return load_generation_config(config_path)


def _discover_generation_stages(generation_dir: Path, logger: EnhancedLogger) -> List[Tuple[str, Path]]:
    """
    Programmatically discover generation stages in order.

    Returns a list of tuples: (stage_name, stage_path)
    Stages are sorted by their numeric prefix (01, 02, etc.)
    """
    stages = []

    if not generation_dir.exists():
        error_msg = f"Generation directory not found: {generation_dir}"
        logger.error(error_msg)
        raise _PipelineConfigurationError(error_msg)

    # Regex pattern to match: digits - name (e.g., "01 - Generate Image", "11 - Final Stage")
    stage_pattern = re.compile(r'^(\d+)\s+-\s+(.+)$')

    for item in generation_dir.iterdir():
        if item.is_dir():
            match = stage_pattern.match(item.name)
            if match:
                stage_num = int(match.group(1))
                stage_name = match.group(2)
                stages.append((stage_name, item, stage_num))

    # Sort by stage number
    stages.sort(key=lambda x: x[2])

    # Return without the stage number
    return [(stage_name, stage_path) for stage_name, stage_path, _ in stages]


def _run_stage(stage_name: str, stage_path: Path, pipeline_log_path: Path,
               generation_config_path: Path, authentication_config_path: Path,
               output_path: Path, logger: EnhancedLogger, debug: bool = False) -> bool:
    """ Run a single generation stage """

    # Find the Python script in the stage directory
    script_files = list(stage_path.glob("*.py"))
    if not script_files:
        logger.error("No Python script found in %s", stage_path)
        return False

    # Use the first Python script found (assuming one main script per stage)
    script_path = script_files[0]
    logger.info("Running script: %s", script_path.name)

    # Construct the command to run
    venv_python = stage_path / "venv" / "bin" / "python"

    if not venv_python.exists():
        logger.error("Virtual environment not found: %s", venv_python)
        return False

    # Build command arguments
    cmd = [
        str(venv_python),
        str(script_path),
        "--config", str(generation_config_path),
        "--authentication", str(authentication_config_path),
        "--output", str(output_path),
        "--log-file", str(pipeline_log_path)
    ]

    if debug:
        logger.debug("Command to execute: %s", ' '.join(cmd))
        logger.debug("Working directory: %s", stage_path)

    try:

        # Use Popen as a context manager to ensure cleanup
        with subprocess.Popen(
            cmd,
            cwd=stage_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        ) as process:
            # Stream output to both console and log
            output_lines = []
            for line in iter(process.stdout.readline, ''):
                if line:
                    # Print to console immediately
                    print(line.rstrip())
                    # Store for logging
                    output_lines.append(line.rstrip())

        # The 'with' block automatically calls process.wait() upon exit.
        # We can now safely check the process's return code.
        if process.returncode == 0:
            logger.info("Stage %s completed successfully", stage_name)
            return True

        logger.error("Stage %s failed with exit code %d", stage_name, process.returncode)
        for line in output_lines:
            logger.error("Stage %s: %s", stage_name, line)
        return False

    except (OSError, FileNotFoundError, PermissionError) as e:
        logger.error("System error running stage %s: %s", stage_name, e)
        return False


def _log_pipeline_start(logger: EnhancedLogger) -> None:
    """ Log the pipeline start with formatted date """
    current_time = datetime.now()
    formatted_time = current_time.strftime("%A %d %B %Y %H:%M")

    logger.info("=" * 60)
    logger.info("=")
    logger.info("= Starting ambient track generation pipeline")
    logger.info("= %s", formatted_time)
    logger.info("=")
    logger.info("=" * 60)
    logger.info("")


def _log_debug_info(logger: EnhancedLogger, generation_config: Generation, pipeline_log_path: Path,
                    generation_config_path: Path, authentication_config_path: Path,
                    output_path: Path) -> None:
    """ Log debug information if debug mode is enabled """
    logger.debug("Generation configuration:")
    logger.debug(generation_config.data.model_dump_json(indent=2))

    logger.debug("Using paths:")
    logger.debug("  Pipeline log: %s", pipeline_log_path)
    logger.debug("  Generation config: %s", generation_config_path)
    logger.debug("  Authentication config: %s", authentication_config_path)
    logger.debug("  Output: %s", output_path)


def _log_stages_found(logger: EnhancedLogger, stages: List[Tuple[str, Path]]) -> None:
    """ Log the discovered generation stages """
    logger.info("")
    logger.info("Found %d generation stages", len(stages))
    for stage_name, stage_path in stages:
        logger.info("  %s (%s)", stage_name, stage_path.name)


def _run_pipeline_stages(logger: EnhancedLogger, stages: List[Tuple[str, Path]], pipeline_log_path: Path,
                         generation_config_path: Path, authentication_config_path: Path,
                         output_path: Path, debug: bool) -> int:
    """ Run all pipeline stages and return count of successful stages """
    successful_stages = 0

    for stage_name, stage_path in stages:
        logger.info("")
        logger.info("=" * 40)
        logger.info("Running stage: %s", stage_name)
        logger.info("=" * 40)

        success = _run_stage(
            stage_name,
            stage_path,
            pipeline_log_path,
            generation_config_path,
            authentication_config_path,
            output_path,
            logger,
            debug
        )

        if success:
            successful_stages += 1
        else:
            logger.error("Stage %s failed", stage_name)
            logger.error("Pipeline stopped due to stage failure")
            break

    return successful_stages


def _log_pipeline_summary(logger: EnhancedLogger, successful_stages: int, total_stages: int) -> bool:
    """ Log the final pipeline execution summary """
    logger.info("")
    logger.info("Pipeline execution completed")
    logger.info("  Successful stages: %d/%d", successful_stages, total_stages)

    logger.info("")
    if successful_stages == total_stages:
        logger.info("All stages completed successfully!")
        return True

    logger.error("Pipeline failed")
    return False


def _archive_run_results(logger: EnhancedLogger, output_path: Path,
                         generation_config_path: Path, pipeline_log_path: Path,
                         generation_config: Generation) -> None:
    """ Archives the results and configuration of a pipeline run into a timestamped folder """
    try:
        run_identifier = datetime.now().strftime("%Y.%m.%d - %H.%M.%S")
        archive_dir = output_path / "output" / run_identifier
        archive_dir.mkdir(parents=True, exist_ok=True)

        logger.info("")
        logger.info("Archiving results to: %s", archive_dir)

        # Copy the contents of the result directory
        source_result_dir = output_path / Paths.RESULT
        if source_result_dir.is_dir():
            dest_result_dir = archive_dir / Paths.RESULT
            shutil.copytree(source_result_dir, dest_result_dir)
            logger.info("  Copied generation results")
        else:
            logger.warning("Run result directory not found, skipping copy: %s", source_result_dir)

        # Copy over the main config
        shutil.copy2(generation_config_path, archive_dir)
        logger.info("  Copied generation config to archive")

        # Create the environment.json file
        # Get Git SHA-1
        git_sha = "unknown"
        try:
            repo_root = Path(__file__).resolve().parent.parent
            git_process = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True, cwd=repo_root
            )
            git_sha = git_process.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning("  Could not get Git SHA: %s. Defaulting to 'unknown'.", e)

        # Write the environment file
        env_data = {
            "sha-1": git_sha,
            "seed": generation_config.data.generation.seed
        }
        env_file_path = archive_dir / "environment.json"
        with open(env_file_path, 'w', encoding='utf-8') as f:
            json.dump(env_data, f, indent=4)
        logger.info("  Created environment file")

        # Finally copy over the log file
        shutil.copy2(pipeline_log_path, archive_dir)
        logger.info("  Copied %s to archive", pipeline_log_path.name)

    except (OSError, shutil.Error, TypeError) as e:
        logger.error("Failed to archive run results due to an I/O error: %s", e)


def main():
    """ Main entry point for the generation pipeline """
    # Set up paths that are needed for logging and archiving
    script_dir = Path(__file__).parent.absolute()
    pipeline_log_path = script_dir / "pipeline.log"
    generation_config_path = script_dir.parent / "generation.json"
    authentication_config_path = script_dir.parent / "authentication.json"
    output_path = script_dir.parent

    # Clean the workspace before starting the run
    result_dir_to_clean = output_path / Paths.RESULT
    _prepare_workspace(pipeline_log_path, result_dir_to_clean)

    # Initialize variables to be available in the finally block
    logger = None
    generation_config = None
    run_was_successful = False

    try:
        # Load configuration then set up our process
        generation_config = _load_and_validate_config(generation_config_path)
        logger = _setup_logging(pipeline_log_path, generation_config.data.debug)

        # Start pipeline
        _log_pipeline_start(logger)

        # Display configuration in debug mode
        if generation_config.data.debug:
            _log_debug_info(logger, generation_config, pipeline_log_path,
                            generation_config_path, authentication_config_path, output_path)

        # Discover generation stages
        generation_dir = script_dir / "generation"
        stages = _discover_generation_stages(generation_dir, logger)

        if not stages:
            raise _PipelineConfigurationError(f"No generation stages found in {generation_dir}")
        _log_stages_found(logger, stages)

        # Run each stage in sequence
        successful_stages = _run_pipeline_stages(
            logger, stages, pipeline_log_path, generation_config_path,
            authentication_config_path, output_path, generation_config.data.debug
        )

        # Final summary
        run_was_successful = _log_pipeline_summary(logger, successful_stages, len(stages))

    except _PipelineConfigurationError as e:
        if logger:
            logger.error("A configuration error stopped the pipeline: %s", e)
        else:
            print(f"A configuration error stopped the pipeline: {e}")
        run_was_successful = False
    except Exception as e:  # pylint: disable=broad-exception-caught
        if logger:
            logger.critical("An unexpected critical error occurred: %s", e, exc_info=True)
        else:
            print(f"An unexpected critical error occurred: {e}")
        run_was_successful = False
    finally:
        # This block will run whether the pipeline succeeds or fails
        if logger and generation_config:
            _archive_run_results(logger, output_path, generation_config_path, pipeline_log_path, generation_config)
        else:
            print("Logger or config not available; unable to perform final archiving.")

        exit_code = 0 if run_was_successful else 1
        if logger:
            logger.info("")
            logger.info("Pipeline finished with exit code %d", exit_code)
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
