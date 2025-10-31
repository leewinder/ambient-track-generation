#!/usr/bin/env python3
""" Batch generation script for processing multiple generations from JSON configuration """

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class PromptModification(BaseModel):
    """ Prompt modification configuration """
    steps: List[str] = Field(description="Steps to modify", min_length=1)
    modifiers: List[str] = Field(description="Modifiers to append to prompts", min_length=1)
    append: bool = Field(description="If True, append modifiers to prompts; if False, prepend")

    @field_validator('modifiers')
    @classmethod
    def validate_modifiers(cls, v: List[str]) -> List[str]:
        """ Ensure no empty strings in modifiers list """
        for modifier in v:
            if not modifier or modifier.strip() == "":
                raise ValueError("Modifiers cannot contain empty strings")
        return v


class ModificationsConfig(BaseModel):
    """ Modifications configuration """
    prompts: List[PromptModification] = Field(description="Prompt modifications", min_length=1)


class BatchConfigModel(BaseModel):
    """ Root batch configuration model """
    steps: Optional[str] = Field(default=None, description="Steps to execute (optional)")
    global_: Dict[str, Any] = Field(default_factory=dict, alias="global", description="Global overrides")
    runs: List[Dict[str, Any]] = Field(description="List of run configurations", min_length=1)
    modifications: Optional[ModificationsConfig] = Field(default=None, description="Prompt modifications")


def _parse_path(path: str) -> List[str]:
    """ Parse dot-notation path, strip leading dot and split by dots """
    if path.startswith('.'):
        path = path[1:]
    return path.split('.') if path else []


def _resolve_path(config: Dict[str, Any], path: str) -> Any:
    """ Navigate to value in nested dict using dot-notation path """
    path_parts = _parse_path(path)
    current = config

    for part in path_parts:
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Path '{path}' not found in configuration")
        current = current[part]

    return current


def _set_path(config: Dict[str, Any], path: str, value: Any) -> None:
    """ Set value at dot-notation path, creating intermediate dicts if needed """
    path_parts = _parse_path(path)
    current = config

    # Navigate to parent, creating intermediate dicts if needed
    for part in path_parts[:-1]:
        if part not in current:
            current[part] = {}
        elif not isinstance(current[part], dict):
            raise ValueError(f"Cannot set path '{path}': '{part}' is not a dict")
        current = current[part]

    # Set the final value
    current[path_parts[-1]] = value


def _path_exists(config: Dict[str, Any], path: str) -> bool:
    """ Check if dot-notation path exists in configuration """
    try:
        _resolve_path(config, path)
        return True
    except KeyError:
        return False


def _validate_paths_exist(config: Dict[str, Any], batch_config: BatchConfigModel) -> List[str]:
    """ Validate all paths in global and runs exist in configuration """
    errors = []

    # Validate global paths
    for path in batch_config.global_.keys():
        if not _path_exists(config, path):
            errors.append(f"Global path '{path}' not found in configuration")

    # Validate run paths
    for i, run in enumerate(batch_config.runs):
        for path in run.keys():
            if not _path_exists(config, path):
                errors.append(f"Run {i+1} path '{path}' not found in configuration")

    return errors


def _validate_prompt_modifications(config: Dict[str, Any], batch_config: BatchConfigModel) -> List[str]:
    """ Validate prompt modifications are valid """
    errors = []

    if not batch_config.modifications:
        return errors

    for i, prompt_mod in enumerate(batch_config.modifications.prompts):
        # Validate each step exists and has prompts field
        for step_name in prompt_mod.steps:
            step_path = f"steps.{step_name}"
            if not _path_exists(config, step_path):
                errors.append(f"Prompt modification {i+1}: step '{step_name}' not found")
                continue

            # Check if step has prompts field
            prompts_path = f"{step_path}.prompts"
            if not _path_exists(config, prompts_path):
                errors.append(f"Prompt modification {i+1}: step '{step_name}' has no prompts field")
                continue

            # Check prompt count matches modifier count
            try:
                prompts = _resolve_path(config, prompts_path)
                if not isinstance(prompts, list):
                    errors.append(f"Prompt modification {i+1}: step '{step_name}' prompts is not a list")
                    continue

                if len(prompts) != len(prompt_mod.modifiers):
                    errors.append(
                        f"Prompt modification {i+1}: step '{step_name}' has {len(prompts)} prompts but {len(prompt_mod.modifiers)} modifiers")
            except KeyError:
                errors.append(f"Prompt modification {i+1}: step '{step_name}' prompts field not accessible")

    return errors


def validate_batch_config(batch_config: BatchConfigModel, config: Dict[str, Any]) -> None:
    """ Validate batch configuration against main configuration """
    errors = []

    # Validate paths exist
    errors.extend(_validate_paths_exist(config, batch_config))

    # Validate prompt modifications
    errors.extend(_validate_prompt_modifications(config, batch_config))

    if errors:
        error_msg = "Batch configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
        raise ValueError(error_msg)


def _apply_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """ Apply overrides to configuration using dot-notation paths """
    for path, value in overrides.items():
        _set_path(config, path, value)


def _apply_prompt_modifications(config: Dict[str, Any], modifications: ModificationsConfig) -> None:
    """ Apply prompt modifications by prepending or appending modifiers to step prompts """
    for prompt_mod in modifications.prompts:
        for step_name in prompt_mod.steps:
            prompts_path = f"steps.{step_name}.prompts"
            prompts = _resolve_path(config, prompts_path)

            # Prepend or append each modifier to corresponding prompt
            for i, modifier in enumerate(prompt_mod.modifiers):
                if i < len(prompts):
                    if prompt_mod.append:
                        prompts[i] = prompts[i] + modifier
                    else:
                        prompts[i] = modifier + prompts[i]


def _backup_configuration(config_path: Path) -> Dict[str, Any]:
    """ Load and return original configuration as backup """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Configuration file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in configuration: {exc}") from exc


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


def write_configuration(config_path: Path, config: Dict[str, Any]) -> None:
    """ Write configuration to file """
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        raise ValueError(f"Failed to write configuration: {exc}") from exc


def execute_runner(steps_arg: Optional[str]) -> None:
    """ Execute generation via generate.sh script with streaming output """
    # Get project root
    project_root = Path(__file__).resolve().parent.parent.parent
    generate_script = project_root / "generate.sh"

    # Validate script exists
    if not generate_script.exists():
        raise FileNotFoundError(f"generate.sh not found: {generate_script}")

    # Build command - default to "all" if steps_arg is None
    cmd = ["/usr/bin/env", "bash", str(generate_script)]
    steps_to_use = steps_arg if steps_arg else "all"
    cmd.extend(['--steps', steps_to_use])

    print(f"[RUNNER] Executing: {' '.join(cmd)}")

    try:
        # Execute subprocess with streaming output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            cwd=str(project_root)
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


def main() -> None:
    """ Main entry point """
    print(f"[BATCH] Starting batch generation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Get script directory and batch config path
        script_dir = Path(__file__).resolve().parent
        batch_config_path = script_dir / "batch.json"
        project_root = script_dir.parent.parent
        config_path = project_root / "configuration.json"

        # Load batch configuration
        print(f"[BATCH] Loading batch configuration from: {batch_config_path}")
        batch_config = load_batch_config(batch_config_path)

        # Load and backup original configuration
        print(f"[BATCH] Loading original configuration from: {config_path}")
        original_config = _backup_configuration(config_path)

        # Validate batch configuration against original config
        print("[BATCH] Validating batch configuration...")
        validate_batch_config(batch_config, original_config)

        print(f"[BATCH] Found {len(batch_config.runs)} runs to process")
        if batch_config.steps:
            print(f"[BATCH] Using steps: {batch_config.steps}")

        # Process each run
        for i, run in enumerate(batch_config.runs, 1):
            run_name = run.get('.name', f'Run {i}')
            print("*" * 60)
            print("*" * 60)
            print("*")
            print(f"* {run_name}")
            print("*")
            print("*" * 60)
            print("*" * 60)

            try:
                # Restore configuration from backup
                config = original_config.copy()

                # Apply global overrides
                if batch_config.global_:
                    print(f"[GENERATION {i}] Applying global overrides...")
                    _apply_overrides(config, batch_config.global_)

                # Apply run-specific overrides
                print(f"[GENERATION {i}] Applying run-specific overrides...")
                _apply_overrides(config, run)

                # Apply prompt modifications
                if batch_config.modifications:
                    print(f"[GENERATION {i}] Applying prompt modifications...")
                    _apply_prompt_modifications(config, batch_config.modifications)

                # Write modified configuration
                print(f"[GENERATION {i}] Writing modified configuration...")
                write_configuration(config_path, config)

                # Execute runner
                print(f"[GENERATION {i}] Executing runner...")
                execute_runner(batch_config.steps)

                print(f"[GENERATION {i}] Completed successfully: {run_name}")

            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                print(f"[GENERATION {i}] Failed: {exc}")
                print("[BATCH] Stopping batch execution due to error")
                raise

        print(f"\n[BATCH] All runs completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[BATCH] Batch generation failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
