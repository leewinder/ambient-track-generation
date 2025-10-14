#!/usr/bin/env python3
""" Batch generation script for processing multiple generations from JSON configuration """

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class BatchGenerationItem(BaseModel):
    """ Single generation item configuration """
    name: str = Field(description="Name for this generation", min_length=1)
    prompts: List[str] = Field(description="Prompts for this generation", min_length=1)

    @field_validator('prompts')
    @classmethod
    def validate_prompts(cls, v: List[str]) -> List[str]:
        """ Ensure no empty strings in prompts list """
        for prompt in v:
            if not prompt or prompt.strip() == "":
                raise ValueError("Prompts cannot contain empty strings")
        return v


class StyleConfig(BaseModel):
    """ Style configuration """
    prompts: List[str] = Field(description="Style prompts to append to generation prompts", min_length=1)

    @field_validator('prompts')
    @classmethod
    def validate_prompts(cls, v: List[str]) -> List[str]:
        """ Ensure no empty strings in prompts list """
        for prompt in v:
            if not prompt or prompt.strip() == "":
                raise ValueError("Style prompts cannot contain empty strings")
        return v


class BatchConfigModel(BaseModel):
    """ Root batch configuration model """
    steps: Optional[str] = Field(default=None, description="Steps to execute (optional)")
    seed: int = Field(description="Seed for generation", gt=0)
    style: StyleConfig = Field(description="Style configuration with prompts")
    generations: List[BatchGenerationItem] = Field(description="List of generations to process", min_length=1)


def load_batch_config(config_path: Path) -> BatchConfigModel:
    """ Load and validate batch configuration JSON """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        return BatchConfigModel(**raw_data)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Batch config file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in batch config: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Batch config validation error: {exc}") from exc


def modify_configuration(config_path: Path, seed: int, name: str, generation_prompts: List[str], style_prompts: List[str]) -> None:
    """ Modify configuration.json with new seed, name, and combined prompts """
    try:
        # Load current configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        # Modify seed
        if 'generation' not in config_data:
            config_data['generation'] = {}
        config_data['generation']['seed'] = seed

        # Modify name
        config_data['name'] = name

        # Combine generation prompts with style prompts
        combined_prompts = []
        for i, gen_prompt in enumerate(generation_prompts):
            if i < len(style_prompts):
                combined_prompt = gen_prompt + style_prompts[i]
                combined_prompts.append(combined_prompt)
            else:
                # If there are more generation prompts than style prompts, use the last style prompt
                combined_prompt = gen_prompt + style_prompts[-1]
                combined_prompts.append(combined_prompt)

        # Modify prompts for all steps that have prompts field
        if 'steps' in config_data:
            for step_config in config_data['steps'].values():
                if 'prompts' in step_config:
                    step_config['prompts'] = combined_prompts

        # Write modified configuration back
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Configuration file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in configuration: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Failed to modify configuration: {exc}") from exc


def execute_runner(steps_arg: Optional[str]) -> None:
    """ Execute runner module via subprocess with streaming output """
    # Get project root
    project_root = Path(__file__).resolve().parent.parent.parent
    runner_dir = project_root / "src" / "modules" / "generation_runner"
    venv_python = runner_dir / "venv" / "bin" / "python"
    main_py = runner_dir / "main.py"

    # Validate paths exist
    if not runner_dir.exists():
        raise FileNotFoundError(f"Runner directory not found: {runner_dir}")
    if not venv_python.exists():
        raise FileNotFoundError(f"Runner venv python not found: {venv_python}")
    if not main_py.exists():
        raise FileNotFoundError(f"Runner main.py not found: {main_py}")

    # Build command
    cmd = [str(venv_python), str(main_py)]
    if steps_arg:
        cmd.extend(['--step', steps_arg])

    print(f"[RUNNER] Executing: {' '.join(cmd)}")

    try:
        # Execute subprocess with streaming output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            cwd=str(runner_dir)
        )

        # Stream output line-by-line to console
        for line in process.stdout:
            print(line, end='')
            sys.stdout.flush()

        # Wait for process to complete
        return_code = process.wait()

        # Check return code
        if return_code != 0:
            raise RuntimeError(f"Runner failed with exit code {return_code}")

        print("[RUNNER] Completed successfully")

    except subprocess.SubprocessError as exc:
        raise RuntimeError(f"Failed to execute runner: {exc}") from exc


def archive_results(name: str) -> None:
    """ Create timestamped archive folder and copy results """
    # Get project root
    project_root = Path(__file__).resolve().parent.parent.parent

    # Create timestamped folder name
    timestamp = datetime.now().strftime("%y.%m.%d-%H.%M.%S")
    archive_folder_name = f"{timestamp} - {name}"
    archive_path = project_root / "archive" / archive_folder_name

    print(f"[ARCHIVE] Creating archive: {archive_path}")

    try:
        # Create archive directory
        archive_path.mkdir(parents=True, exist_ok=True)

        # Copy entire output folder contents
        output_path = project_root / "output"
        if output_path.exists():
            for item in output_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, archive_path / item.name)
                elif item.is_dir():
                    shutil.copytree(item, archive_path / item.name)

        # Copy configuration.json
        config_path = project_root / "configuration.json"
        if config_path.exists():
            shutil.copy2(config_path, archive_path / "configuration.json")

        # Copy runner's pipeline.log
        runner_log_path = project_root / "src" / "modules" / "generation_runner" / "pipeline.log"
        if runner_log_path.exists():
            shutil.copy2(runner_log_path, archive_path / "pipeline.log")

        print(f"[ARCHIVE] Archive created successfully: {archive_path}")

    except Exception as exc:
        raise RuntimeError(f"Failed to create archive: {exc}") from exc


def main() -> None:
    """ Main entry point """
    print(f"[BATCH] Starting batch generation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Get script directory and batch config path
        script_dir = Path(__file__).resolve().parent
        batch_config_path = script_dir / "batch.json"
        project_root = script_dir.parent.parent

        # Load batch configuration
        print(f"[BATCH] Loading batch configuration from: {batch_config_path}")
        batch_config = load_batch_config(batch_config_path)

        print(f"[BATCH] Found {len(batch_config.generations)} generations to process")
        print(f"[BATCH] Using seed: {batch_config.seed}")
        if batch_config.steps:
            print(f"[BATCH] Using steps: {batch_config.steps}")

        # Process each generation
        for i, generation in enumerate(batch_config.generations, 1):
            print("*" * 60)
            print("*" * 60)
            print("*")
            print(f"* {generation.name}")
            print("*")
            print("*" * 60)
            print("*" * 60)

            try:
                # Modify configuration
                config_path = project_root / "configuration.json"
                print(f"[GENERATION {i}] Modifying configuration...")
                style_prompts = getattr(batch_config.style, 'prompts')
                modify_configuration(config_path, batch_config.seed, generation.name, generation.prompts, style_prompts)

                # Execute runner
                print(f"[GENERATION {i}] Executing runner...")
                execute_runner(batch_config.steps)

                # Archive results
                print(f"[GENERATION {i}] Archiving results...")
                archive_results(generation.name)

                print(f"[GENERATION {i}] Completed successfully: {generation.name}")

            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                print(f"[GENERATION {i}] Failed: {exc}")
                print("[BATCH] Stopping batch execution due to error")
                raise

        print(f"\n[BATCH] All generations completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[BATCH] Batch generation failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
