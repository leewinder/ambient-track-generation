#!/usr/bin/env python3
""" Archive video module for archiving generated video production """

import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List

from pipeline_utilities.args import parse_arguments
from pipeline_utilities.configuration import load_configuration
from pipeline_utilities.logs import EnhancedLogger
from pipeline_utilities.paths import Paths, Project


class VideoArchiver:
    """ Main video archiver orchestrator """

    def __init__(self, logger: EnhancedLogger):
        """ Initialize the video archiver """
        self.logger = logger

    def _generate_timestamp(self) -> tuple[str, str, str]:
        """ Generate timestamp components for archive folder name and metadata """
        now = datetime.now()
        date_str = now.strftime("%y.%m.%d")
        hour_min = now.strftime("%H.%M")
        seconds = now.strftime("%S")
        time_str_metadata = f"{hour_min}:{seconds}"
        time_str_folder = now.strftime("%H.%M.%S")
        return date_str, time_str_metadata, f"{date_str}-{time_str_folder}"

    def _create_archive_folder(self, archive_name: str) -> Path:
        """ Create archive folder at project root archive directory """
        archive_folder = Project.get_root() / "archive" / archive_name
        archive_folder.mkdir(parents=True, exist_ok=True)
        return archive_folder

    def _match_files(self, pattern: str, directory: Path) -> List[Path]:
        """ Match files in directory against regex pattern (supports patterns with ^ and $ anchors) """
        matched_files = []
        regex = re.compile(pattern)

        if not directory.exists():
            self.logger.warning(f"Directory does not exist: {directory}")
            return matched_files

        for file_path in directory.iterdir():
            if file_path.is_file() and regex.search(file_path.name):
                matched_files.append(file_path)

        return sorted(matched_files)

    def _copy_files(self, source_files: List[Path], dest_dir: Path, file_type: str) -> None:
        """ Copy files to destination directory """
        if not source_files:
            self.logger.info(f"No {file_type} files to copy")
            return

        dest_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Copying {len(source_files)} {file_type} file(s)")

        for source_file in source_files:
            if not source_file.exists():
                self.logger.warning(f"Source file not found: {source_file}")
                continue

            dest_file = dest_dir / source_file.name
            shutil.copy2(source_file, dest_file)
            self.logger.debug(f"Copied: {source_file.name}")

    def _create_metadata(self, archive_folder: Path, date_str: str, time_str: str) -> None:
        """ Create metadata.json in archive folder """
        metadata = {
            "date": date_str,
            "time": time_str
        }

        metadata_path = archive_folder / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info("Created metadata.json")

    def _copy_configuration_files(self, archive_folder: Path) -> None:
        """ Copy configuration.json and pipeline.log to archive """
        config_dir = archive_folder / "configuration"
        config_dir.mkdir(parents=True, exist_ok=True)

        project_root = Project.get_root()

        # Copy configuration.json
        config_source = project_root / "configuration.json"
        if config_source.exists():
            shutil.copy2(config_source, config_dir / "configuration.json")
            self.logger.debug("Copied configuration.json")
        else:
            self.logger.warning("configuration.json not found, skipping")

        # Copy pipeline.log
        log_source = project_root / "pipeline.log"
        if log_source.exists():
            shutil.copy2(log_source, config_dir / "pipeline.log")
            self.logger.debug("Copied pipeline.log")
        else:
            self.logger.warning("pipeline.log not found, skipping")

    def execute_step(self, step_name: str, config_data: Any) -> None:
        """ Execute a single archive video step from the configuration """
        start_time = time.time()

        self.logger.header(f"Executing step: {step_name}")

        # Get step configuration
        step_config = config_data.steps.get(step_name)
        if not step_config:
            raise ValueError(f"Step '{step_name}' not found in configuration")

        # Validate that this is an ArchiveVideoStep
        if not hasattr(step_config, 'audio') or not hasattr(step_config, 'artwork') or not hasattr(step_config, 'video'):
            raise ValueError(f"Step '{step_name}' is not configured for video archiving")

        self.logger.info(f"Step: {step_name}")
        self.logger.info(f"Module: {step_config.module}")
        self.logger.info(f"Audio pattern: {step_config.audio}")
        self.logger.info(f"Artwork pattern: {step_config.artwork}")
        self.logger.info(f"Video: {step_config.video}")

        # Generate timestamp and create archive folder
        date_str, time_str, timestamp = self._generate_timestamp()
        archive_name = f"{timestamp} - {config_data.name}"

        self.logger.info(f"Creating archive folder: archive/{archive_name}")
        archive_folder = self._create_archive_folder(archive_name)

        # Get interim directory
        interim_dir = Project.get_root() / Paths.OUTPUT / Paths.INTERIM

        # Match and copy audio files
        self.logger.info("Matching audio files")
        audio_files = self._match_files(step_config.audio, interim_dir)
        if audio_files:
            audio_dir = archive_folder / "audio"
            self._copy_files(audio_files, audio_dir, "audio")
        else:
            self.logger.info("No audio files to copy")

        # Match and copy artwork files
        self.logger.info("Matching artwork files")
        artwork_files = self._match_files(step_config.artwork, interim_dir)
        if artwork_files:
            artwork_dir = archive_folder / "artwork"
            self._copy_files(artwork_files, artwork_dir, "artwork")
        else:
            self.logger.info("No artwork files to copy")

        # Copy video file
        video_source = interim_dir / step_config.video
        if video_source.exists():
            self.logger.info(f"Copying video file: {step_config.video}")
            shutil.copy2(video_source, archive_folder / step_config.video)
        else:
            self.logger.warning(f"Video file not found: {step_config.video}")

        # Create metadata.json
        self.logger.info("Creating metadata")
        self._create_metadata(archive_folder, date_str, time_str)

        # Copy configuration files
        self.logger.info("Copying configuration files")
        self._copy_configuration_files(archive_folder)

        # Log completion
        duration = time.time() - start_time
        self.logger.header("Step completed successfully")
        self.logger.info(f"Step name: {step_name}")
        self.logger.info(f"Module: {step_config.module}")
        self.logger.info(f"Archive folder: archive/{archive_name}")
        self.logger.info(f"Audio files: {len(audio_files)}")
        self.logger.info(f"Artwork files: {len(artwork_files)}")
        self.logger.info(f"Duration: {duration:.2f} seconds")


def main() -> None:
    """ Main entry point """
    # Parse arguments
    args = parse_arguments("Archive video module for archiving generated video production")

    try:
        # Load configuration first to get debug setting
        config_path = Project.get_configuration()
        config_loader = load_configuration(config_path)
        config_data = config_loader.data

        # Setup logging with debug setting from config
        logger = EnhancedLogger.setup_pipeline_logging(
            log_file=args.log_file,
            debug=config_data.debug or False,
            script_name="archive_video"
        )

        logger.info(f"Loaded configuration: {config_data.name}")

        # Initialize video archiver
        archiver = VideoArchiver(logger)

        # Execute the step
        archiver.execute_step(args.step, config_data)

    except Exception as exc:
        logger.error(f"Archive video failed: {exc}")
        raise


if __name__ == "__main__":
    main()
