#!/usr/bin/env python3
"""
Simple batch generation script that processes all prompts from prompts.json

This script:
1. Reads prompts.json to get all prompt entries from the config array
2. For each entry, copies the values to generation.json
3. Runs the generation process via generate.sh

Max stages limit is controlled by the max_generation_steps field in prompts.json
"""

import json
import subprocess
import sys
import argparse
from pathlib import Path

HIGHLIGHT_CHARACTER_LENGTH = 120
HIGHLIGHT_CHARACTER = "*"


def print_heavy_border() -> None:
    """ Print a heavy 6-line star border """
    for _ in range(6):
        print(HIGHLIGHT_CHARACTER * HIGHLIGHT_CHARACTER_LENGTH)


def print_light_border() -> None:
    """ Print a light 3-line star border """
    for _ in range(3):
        print(HIGHLIGHT_CHARACTER * HIGHLIGHT_CHARACTER_LENGTH)


def print_section_header(title: str) -> None:
    """ Print a formatted section header with equals borders """
    print("=" * 60)
    print(f"= {title}")
    print("=" * 60)


def print_generation_spacing() -> None:
    """ Print spacing before generation steps """
    print("\n" * 10)


def parse_arguments() -> argparse.Namespace:
    """ Parse command line arguments for the batch generation script """
    parser = argparse.ArgumentParser(
        description="Simple batch generation script that processes all prompts from prompts.json"
    )

    parser.add_argument(
        '--create-generation-script',
        action='store_true',
        help='Create generation.json from a specific config/build without running generation'
    )

    parser.add_argument(
        '--config',
        type=int,
        help='Config index (0-indexed) to use when --create-generation-script is specified'
    )

    parser.add_argument(
        '--build',
        type=int,
        help='Build index (0-indexed) to use when --create-generation-script is specified'
    )

    return parser.parse_args()


def get_project_root() -> Path:
    """ Get the project root directory by finding the generation.json file """
    current_dir = Path.cwd()

    # Walk up the directory tree to find generation.json
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "generation.json").exists():
            return parent

    raise FileNotFoundError("Could not find generation.json file. Please run from project directory or subdirectory.")


def load_json_file(file_path: Path) -> dict:
    """ Load and parse a JSON file """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR: Failed to load {file_path}: {e}")
        sys.exit(1)


def save_json_file(file_path: Path, data: dict) -> None:
    """ Save data to a JSON file """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"ERROR: Failed to save {file_path}: {e}")
        sys.exit(1)


def merge_common_and_build(common: dict, build: dict) -> dict:
    """ Merge common and build.properties, with build.properties taking precedence """
    # Start with common as base
    merged = common.copy()

    # Get the properties from the build
    build_properties = build.get("properties", {})

    # Deep merge build properties, with build properties taking precedence
    merged = _deep_merge(merged, build_properties)

    # Special handling for name field - combine as "common_name (build_name)"
    if "name" in common and "name" in build_properties:
        merged["name"] = f"{common['name']} ({build_properties['name']})"
    elif "name" in common:
        merged["name"] = common["name"]
    elif "name" in build_properties:
        merged["name"] = build_properties["name"]

    return merged


def _deep_merge(base: dict, override: dict) -> dict:
    """ Deep merge two dictionaries, with override taking precedence """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = _deep_merge(result[key], value)
        else:
            # Override with new value
            result[key] = value

    return result


def validate_config_build_indices(prompts_data: dict, config_idx: int, build_idx: int) -> None:
    """ Validate that the specified config and build indices exist """
    config_array = prompts_data.get("config", [])

    if not config_array:
        raise ValueError("No 'config' array found in prompts.json")

    if config_idx < 0 or config_idx >= len(config_array):
        raise ValueError(f"Config index {config_idx} is out of range. Available configs: 0-{len(config_array)-1}")

    config_entry = config_array[config_idx]
    builds = config_entry.get("builds", [])

    if not builds:
        raise ValueError(f"No builds found in config {config_idx}")

    if build_idx < 0 or build_idx >= len(builds):
        raise ValueError(
            f"Build index {build_idx} is out of range for config {config_idx}. Available builds: 0-{len(builds)-1}")


def create_generation_file(project_root: Path, prompts_data: dict, config_idx: int, build_idx: int) -> None:
    """ Create generation.json from a specific config/build without running generation """
    print_section_header("Create Generation File")
    print()

    # Validate indices
    print(f"Validating config {config_idx}, build {build_idx}...")
    validate_config_build_indices(prompts_data, config_idx, build_idx)
    print("✓ Validation passed")

    # Get the specific config and build
    config_entry = prompts_data["config"][config_idx]
    common = config_entry.get("common", {})
    build = config_entry["builds"][build_idx]
    build_properties = build.get("properties", {})

    print(f"Config: {common.get('name', 'Unnamed')}")
    print(f"Build: {build_properties.get('name', 'Unnamed')}")
    print()

    # Merge common and build properties
    print("Merging common and build properties...")
    merged_config = merge_common_and_build(common, build)
    entry_name = merged_config.get("name", f"Config {config_idx} Build {build_idx}")
    print(f"✓ Merged configuration: {entry_name}")

    # Save to generation.json
    generation_file = project_root / "generation.json"
    print(f"Saving to: {generation_file}")
    save_json_file(generation_file, merged_config)
    print("✓ Generation file created successfully!")
    print()
    print_section_header("Complete")


def run_generation(project_root: Path, max_stages: int | None = None) -> int:
    """ Run the generation script and return the exit code """
    generate_script = project_root / "generate.sh"

    if not generate_script.exists():
        print(f"ERROR: generate.sh not found at {generate_script}")
        return 1

    try:
        # Build command with optional max_stages parameter
        cmd = ["bash", str(generate_script)]
        if max_stages is not None:
            cmd.extend(["--max-stages", str(max_stages)])

        # Run generate.sh and inherit stdout/stderr to show all output
        process = subprocess.Popen(
            cmd,
            cwd=project_root,
            stdout=None,  # Inherit stdout
            stderr=None,  # Inherit stderr
            text=True
        )

        # Wait for process to complete and get return code
        return_code = process.wait()
        return return_code

    except (OSError, subprocess.SubprocessError) as e:
        print(f"ERROR: Failed to run generation script: {e}")
        return 1


def main():
    """ Main execution function """
    # Parse command line arguments
    args = parse_arguments()

    # Get project root directory
    try:
        project_root = get_project_root()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Load prompts data
    prompts_file = Path(__file__).parent / "prompts.json"
    if not prompts_file.exists():
        print(f"ERROR: prompts.json not found at {prompts_file}")
        sys.exit(1)

    print(f"Loading prompts from: {prompts_file}")
    prompts_data = load_json_file(prompts_file)

    # Handle create-generation-script mode
    if args.create_generation_script:
        if args.config is None or args.build is None:
            print("ERROR: --config and --build are required when using --create-generation-script")
            print("Usage: python batch_generate.py --create-generation-script --config 0 --build 1")
            sys.exit(1)

        try:
            create_generation_file(project_root, prompts_data, args.config, args.build)
            return
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

    # Regular batch generation mode
    print_section_header("Simple Batch Generation Script")
    print()

    print(f"Project root: {project_root}")

    # Check for generation.json (only needed for batch mode)
    generation_file = project_root / "generation.json"
    if not generation_file.exists():
        print(f"ERROR: generation.json not found at {generation_file}")
        sys.exit(1)

    # Get the config array from prompts.json
    config_array = prompts_data.get("config", [])
    if not config_array:
        print("ERROR: No 'config' array found in prompts.json")
        sys.exit(1)

    # Count total builds across all configs
    total_builds = 0
    for config_entry in config_array:
        builds = config_entry.get("builds", [])
        for build in builds:
            # Get number_of_images, default to 1 if not present
            number_of_images = build.get("number_of_images", 1)
            if not isinstance(number_of_images, int) or number_of_images < 1:
                print(f"ERROR: Invalid number_of_images in build: {number_of_images}. Must be >= 1")
                sys.exit(1)
            total_builds += number_of_images

    # Determine max_stages from config file
    max_stages = None
    if "max_generation_steps" in prompts_data:
        max_stages = prompts_data["max_generation_steps"]
        total_stages = max_stages
    else:
        total_stages = "All"  # No limit - run all available stages

    # Print the prominent header
    print_generation_spacing()
    print_heavy_border()
    print("")
    print("                    STARTING BATCH GENERATION")
    print("")
    print(f"                    Total Images To Generate: {total_builds}")
    print(f"                    Number Of Configurations: {len(config_array)}")
    print(f"                    Stages To Run: {total_stages}")
    print("")
    print_heavy_border()
    print("\n" * 2)

    # Process each configuration entry
    successful_runs = 0
    failed_runs = 0
    current_build = 0

    for config_idx, config_entry in enumerate(config_array, 1):
        common = config_entry.get("common", {})
        builds = config_entry.get("builds", [])

        if not common:
            print(f"ERROR: Missing 'common' section in config {config_idx}")
            continue

        if not builds:
            print(f"ERROR: No builds found in config {config_idx}")
            continue

        # Count total builds in this config
        config_total_builds = 0
        for build in builds:
            number_of_images = build.get("number_of_images", 1)
            config_total_builds += number_of_images

        print(f"Processing config group {config_idx} with {len(builds)} builds")
        print(f"Common name: {common.get('name', 'Unnamed')}")
        print(f"Total builds in this config: {config_total_builds}")
        print()

        # Process each build in this config
        current_config_build = 0
        for build_idx, build_entry in enumerate(builds, 1):
            # Get number_of_images, default to 1 if not present
            number_of_images = build_entry.get("number_of_images", 1)
            if not isinstance(number_of_images, int) or number_of_images < 1:
                print(f"ERROR: Invalid number_of_images in build {build_idx}: {number_of_images}. Must be >= 1")
                continue

            # Merge common and build properties
            merged_config = merge_common_and_build(common, build_entry)
            base_name = merged_config.get("name", f"Config {config_idx} Build {build_idx}")

            # Process each run of this build
            for run_idx in range(1, number_of_images + 1):
                current_build += 1
                current_config_build += 1

                # Create the run-specific name
                entry_name = f"{base_name} ({run_idx} of {number_of_images})"

                # Update the merged config with the run-specific name
                merged_config["name"] = entry_name

                # Add lots of spacing and clear visual separator for new generation
                print_generation_spacing()
                print_light_border()
                print("")
                print("                    STARTING A NEW GENERATION STEP")
                print("")
                print(
                    f"                    Build {current_build}/{total_builds} : Config {current_config_build}/{config_total_builds} : {entry_name}")
                print("")
                print_light_border()
                print("\n")

                # Save merged configuration to generation.json
                print("Updating generation.json with configuration...")
                save_json_file(generation_file, merged_config)

                # Run generation
                print("Starting generation process...")
                print()

                exit_code = run_generation(project_root, max_stages)

                if exit_code == 0:
                    print(f"✓ Generation completed successfully for '{entry_name}'")
                    successful_runs += 1
                else:
                    print(f"✗ Generation failed for '{entry_name}' (exit code: {exit_code})")
                    failed_runs += 1

                print()

    # Final summary
    print_section_header("Batch Generation Complete")
    print(f"Total builds processed: {total_builds}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")

    if failed_runs > 0:
        print(f"{failed_runs} generation(s) failed")
        sys.exit(1)
    else:
        print("All generations completed successfully!")


if __name__ == "__main__":
    main()
