#!/usr/bin/env python3
""" Video construction module for creating videos from static images and audio """

import time
from pathlib import Path
from typing import Any

import ffmpeg

from pipeline_utilities.args import parse_arguments
from pipeline_utilities.configuration import load_configuration
from pipeline_utilities.logs import EnhancedLogger
from pipeline_utilities.paths import Paths, Project


class VideoConstructor:
    """ Main video constructor orchestrator """

    def __init__(self, logger: EnhancedLogger):
        """ Initialize the video constructor """
        self.logger = logger

    def _validate_inputs(self, image_path: Path, audio_path: Path, video_length_seconds: float) -> None:
        """ Check file existence and audio duration >= video length """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio_duration = self._get_audio_duration(audio_path)
        if audio_duration < video_length_seconds:
            raise ValueError(
                f"Audio duration ({audio_duration:.2f}s) is shorter than required video length "
                f"({video_length_seconds:.2f}s)"
            )

        self.logger.info("Input validation passed:")
        self.logger.info(f"  Image: {image_path.name}")
        self.logger.info(f"  Audio: {audio_path.name} ({audio_duration:.2f}s)")

    def _validate_output_format(self, output_filename: str) -> None:
        """ Extract extension and verify basic format requirements """
        output_path = Path(output_filename)
        extension = output_path.suffix.lower()

        if not extension:
            raise ValueError("Output filename must have a file extension")

        # Basic validation - ffmpeg-python will handle actual format support validation
        # when we try to create the video
        self.logger.info(f"Output format: {extension}")

    def _get_audio_duration(self, audio_path: Path) -> float:
        """ Use ffmpeg.probe to get audio length """
        try:
            probe = ffmpeg.probe(str(audio_path))
            duration = float(probe['format']['duration'])
            return duration
        except ffmpeg.Error as exc:
            raise ValueError(f"Failed to probe audio file {audio_path}: {exc}") from exc

    def _create_video(self, image_path: Path, audio_path: Path, output_path: Path, video_length_seconds: float) -> None:
        """ Use ffmpeg-python to create video with static image and audio """
        self.logger.info(f"Creating video: {output_path.name}")
        self.logger.info(f"Video length: {video_length_seconds:.2f} seconds")

        try:
            # Optimize for static image: very low frame rate + stillimage tuning
            frame_rate = 1 / 60  # 1 frame per minute (60 seconds) - for 60 min = 60 frames

            self.logger.info(f"Creating video from static image with audio at {frame_rate:.3f} fps...")

            # Video stream from static image for full duration
            video_stream = (
                ffmpeg
                .input(str(image_path), t=video_length_seconds, framerate=frame_rate)
                .video
            )

            # Audio stream
            audio_stream = (
                ffmpeg
                .input(str(audio_path))
                .audio
            )

            # Combine in single output with stillimage optimization
            output_stream = ffmpeg.output(
                video_stream,
                audio_stream,
                str(output_path),
                vcodec='libx264',
                acodec='aac',
                pix_fmt='yuv420p',
                tune='stillimage',  # Optimize H.264 for static image
                t=video_length_seconds
            )

            ffmpeg.run(output_stream, overwrite_output=True, quiet=True)

            self.logger.info(f"Video created successfully: {output_path.name}")

        except ffmpeg.Error as exc:
            raise RuntimeError(f"ffmpeg execution failed: {exc}") from exc

    def _execute_step(self, step_name: str, config_data: Any) -> None:
        """ Execute a single video construction step from the configuration """
        start_time = time.time()

        self.logger.header(f"Executing step: {step_name}")

        # Get step configuration
        step_config = config_data.steps.get(step_name)
        if not step_config:
            raise ValueError(f"Step '{step_name}' not found in configuration")

        # Validate that this is a VideoConstructionStep
        if not hasattr(step_config, 'image') or not hasattr(step_config, 'audio'):
            raise ValueError(f"Step '{step_name}' is not configured for video construction")

        # Validate video configuration exists
        if not config_data.video:
            raise ValueError("Video configuration is required for video construction")

        self.logger.info(f"Step: {step_name}")
        self.logger.info(f"Module: {step_config.module}")
        self.logger.info(f"Image: {step_config.image}")
        self.logger.info(f"Audio: {step_config.audio}")
        self.logger.info(f"Output: {step_config.output}")

        # Get file paths
        image_path = Paths.get_interim_path(step_config.image)
        audio_path = Paths.get_interim_path(step_config.audio)
        output_path = Paths.get_interim_path(step_config.output)

        # Convert video length from minutes to seconds
        video_length_seconds = config_data.video.length * 60

        # Validate inputs
        self.logger.info("Validating inputs")
        self._validate_inputs(image_path, audio_path, video_length_seconds)

        # Validate output format
        self.logger.info("Validating output format")
        self._validate_output_format(step_config.output)

        # Create video
        self.logger.info("Creating video")
        self._create_video(image_path, audio_path, output_path, video_length_seconds)

        # Log completion
        duration = time.time() - start_time
        self.logger.header("Step completed successfully")
        self.logger.info(f"Step name: {step_name}")
        self.logger.info(f"Module: {step_config.module}")
        self.logger.info(f"Output file: {step_config.output}")
        self.logger.info(f"Video length: {config_data.video.length} minutes")
        self.logger.info(f"Duration: {duration:.2f} seconds")


def main() -> None:
    """ Main entry point """
    # Parse arguments
    args = parse_arguments("Video construction module for creating videos from images and audio")

    try:
        # Load configuration first to get debug setting
        config_path = Project.get_configuration()
        config_loader = load_configuration(config_path)
        config_data = config_loader.data

        # Setup logging with debug setting from config
        logger = EnhancedLogger.setup_pipeline_logging(
            log_file=args.log_file,
            debug=config_data.debug or False,
            script_name="video_construction"
        )

        logger.info(f"Loaded configuration: {config_data.name}")

        # Initialize video constructor
        constructor = VideoConstructor(logger)

        # Execute the step
        constructor._execute_step(args.step, config_data)

    except Exception as exc:
        logger.error(f"Video construction failed: {exc}")
        raise


if __name__ == "__main__":
    main()
