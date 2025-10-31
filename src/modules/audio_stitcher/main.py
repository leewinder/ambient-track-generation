#!/usr/bin/env python3
""" Audio stitcher module for combining multiple audio samples with normalization and cross-fading """

import math
import re
import subprocess
import time
from pathlib import Path
from typing import Any, List

from pydub import AudioSegment
from pydub.effects import normalize

from pipeline_utilities.args import parse_arguments
from pipeline_utilities.configuration import load_configuration
from pipeline_utilities.logs import EnhancedLogger
from pipeline_utilities.paths import Paths, Project


class AudioStitcher:
    """ Main audio stitcher orchestrator """

    def __init__(self, logger: EnhancedLogger):
        """ Initialize the audio stitcher """
        self.logger = logger

    def identify_samples(self, input_filename: str) -> List[Path]:
        """ Find all files matching the input pattern and sort them alphabetically """
        # Extract stem from input filename (e.g., "04_audio.mp3" -> "04_audio")
        input_path = Path(input_filename)
        stem = input_path.stem

        # Create regex pattern to match exactly: {stem}_pass_{nnn}.mp3
        # This ensures we only get original pass files, not normalized or intermediate ones
        pattern = re.compile(rf"^{re.escape(stem)}_pass_\d+\.mp3$")

        # Get interim directory path
        interim_dir = Project.get_root() / Paths.OUTPUT / Paths.INTERIM

        # Find all matching files using regex
        matching_files = []
        for file_path in interim_dir.glob("*.mp3"):
            if pattern.match(file_path.name):
                matching_files.append(file_path)

        if not matching_files:
            raise FileNotFoundError(f"No audio pass files found matching pattern: {stem}_pass_*.mp3")

        # Sort alphabetically (ascending order)
        matching_files.sort()

        self.logger.info(f"Found {len(matching_files)} audio samples")
        for file_path in matching_files:
            self.logger.debug(f"  {file_path.name}")

        return matching_files

    def normalize_samples(self, samples: List[Path], headroom: float) -> List[Path]:
        """ Normalize all samples to the same volume level """
        normalized_samples = []

        self.logger.info(f"Normalizing {len(samples)} samples with {headroom}dB headroom")

        for sample_path in samples:
            self.logger.info(f"Normalizing: {sample_path.name}")

            # Load audio file
            audio = AudioSegment.from_mp3(str(sample_path))

            # Apply normalization
            normalized_audio = normalize(audio, headroom=headroom)

            # Create output filename
            output_path = sample_path.parent / f"{sample_path.stem}_normalised{sample_path.suffix}"

            # Export at 256 kbps
            normalized_audio.export(
                str(output_path),
                format="mp3",
                bitrate="256k"
            )

            normalized_samples.append(output_path)
            self.logger.debug(f"Saved normalized sample: {output_path.name}")

        return normalized_samples

    def stitch_samples(self, normalized_samples: List[Path],
                       stitch_fade_duration: float, output_filename: str) -> None:
        """ Stitch normalized samples together with cross-fading """
        if not normalized_samples:
            raise ValueError("No normalized samples provided for stitching")

        self.logger.info(f"Stitching {len(normalized_samples)} samples with {stitch_fade_duration}s fade")

        # Convert fade duration to milliseconds
        fade_ms = int(stitch_fade_duration * 1000)

        # Load first sample as base
        result = AudioSegment.from_mp3(str(normalized_samples[0]))
        self.logger.debug(f"Starting with: {normalized_samples[0].name}")

        # Append remaining samples with cross-fade
        for i, sample_path in enumerate(normalized_samples[1:], 1):
            self.logger.debug(f"Adding sample {i+1}/{len(normalized_samples)}: {sample_path.name}")

            sample = AudioSegment.from_mp3(str(sample_path))
            result = result.append(sample, crossfade=fade_ms)

        # Get output path
        output_path = Paths.get_interim_path(output_filename)

        # Export final result at 256 kbps
        result.export(
            str(output_path),
            format="mp3",
            bitrate="256k"
        )

        self.logger.info(f"Stitched audio saved: {output_filename}")
        self.logger.info(f"Final duration: {len(result) / 1000:.2f} seconds")

    def loop_to_video_length(self, stitched_sample_path: Path, config_data: Any,
                             step_config: Any) -> None:
        """ Loop the stitched sample to match video length and apply intro/outro fades """
        # Validate video configuration exists
        if not config_data.video:
            raise ValueError("Video configuration is required for audio stitching")

        # Calculate target duration in milliseconds
        target_duration_minutes = config_data.video.length
        target_duration_ms = int(target_duration_minutes * 60 * 1000)

        self.logger.info(f"Target video duration: {target_duration_minutes} minutes ({target_duration_ms}ms)")

        # Load the stitched sample
        stitched_sample = AudioSegment.from_mp3(str(stitched_sample_path))
        sample_duration_ms = len(stitched_sample)

        self.logger.info(f"Stitched sample duration: {sample_duration_ms / 1000:.2f} seconds")

        # Calculate how many loops we need
        loops_needed = math.ceil(target_duration_ms / sample_duration_ms)
        self.logger.debug(f"Need {loops_needed} loops to reach target duration")

        # Start with the first sample
        result = stitched_sample
        self.logger.debug("Starting loop 1")

        # Add additional loops with cross-fade
        stitch_fade_ms = int(step_config.stitch_fade * 1000)

        for loop_num in range(2, loops_needed + 1):
            self.logger.debug(f"Adding loop {loop_num}/{loops_needed}")
            result = result.append(stitched_sample, crossfade=stitch_fade_ms)

        # Trim to exact target duration
        if len(result) > target_duration_ms:
            self.logger.debug(f"Trimming from {len(result) / 1000:.2f}s to {target_duration_ms / 1000:.2f}s")
            result = result[:target_duration_ms]

        # Apply intro fade-in
        intro_fade_ms = int(step_config.intro_fade * 1000)
        if intro_fade_ms > 0:
            self.logger.debug(f"Applying {step_config.intro_fade}s intro fade-in")
            result = result.fade_in(intro_fade_ms)

        # Apply outro fade-out
        outro_fade_ms = int(step_config.outro_fade * 1000)
        if outro_fade_ms > 0:
            self.logger.debug(f"Applying {step_config.outro_fade}s outro fade-out")
            result = result.fade_out(outro_fade_ms)

        # Get final output path
        output_path = Paths.get_interim_path(step_config.output)

        # Export final result at 256 kbps using FFmpeg to bypass pydub's WAV limitation
        # for large audio files (>4GB uncompressed)
        self.export_with_ffmpeg(
            result,
            str(output_path),
            "256k"
        )

        self.logger.info(f"Final audio saved: {step_config.output}")
        self.logger.info(f"Final duration: {len(result) / 1000:.2f} seconds ({len(result) / 60000:.2f} minutes)")

    def cleanup_intermediate_files(self, normalized_samples: List[Path],
                                   intermediate_output: str) -> None:
        """ Clean up intermediate files after successful processing """
        self.logger.info("Cleaning up intermediate files")

        # Remove normalized files
        for normalized_file in normalized_samples:
            if normalized_file.exists():
                normalized_file.unlink()
                self.logger.debug(f"Removed: {normalized_file.name}")

        # Remove intermediate stitched file
        intermediate_path = Paths.get_interim_path(intermediate_output)
        if intermediate_path.exists():
            intermediate_path.unlink()
            self.logger.debug(f"Removed: {intermediate_output}")

    def _get_ffmpeg_format_string(self, segment: AudioSegment) -> str:
        """ Map AudioSegment sample width to FFmpeg format string """
        if segment.sample_width == 1:
            return 'u8'  # 8-bit unsigned
        elif segment.sample_width == 2:
            return 's16le'  # 16-bit signed little-endian
        elif segment.sample_width == 4:
            return 's32le'  # 32-bit signed little-endian
        else:
            raise ValueError(f"Unsupported sample width: {segment.sample_width}")

    def _time_to_seconds(self, time_str: str) -> float:
        """ Convert FFmpeg time string (HH:MM:SS.ms) to seconds """
        try:
            parts = time_str.split(':')
            if len(parts) != 3:
                return 0.0
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        except (ValueError, IndexError):
            return 0.0

    def _monitor_progress(self, stderr_fd, total_duration_seconds: float) -> None:
        """ Monitor FFmpeg progress and log percentage complete """
        last_logged_percent = -1

        while True:
            line = stderr_fd.readline()
            if not line:
                break

            line_str = line.decode('utf-8', errors='ignore')

            # Look for time progress
            if 'time=' in line_str:
                # Format: time=00:03:45.23
                try:
                    parts = line_str.split('time=')
                    if len(parts) > 1:
                        time_str = parts[1].split()[0]
                        current_seconds = self._time_to_seconds(time_str)
                        percent = (current_seconds / total_duration_seconds) * 100 if total_duration_seconds > 0 else 0

                        # Only log every 1% to avoid spam
                        percent_int = int(percent)
                        if percent_int != last_logged_percent and percent_int >= 0 and percent_int <= 100:
                            last_logged_percent = percent_int
                            self.logger.debug(f"Encoding progress: {percent:.1f}%")
                except (IndexError, ValueError, AttributeError):
                    pass

    def export_with_ffmpeg(self, segment: AudioSegment, output_path: str, bitrate: str) -> None:
        """ Export AudioSegment directly to MP3 using FFmpeg, bypassing pydub's WAV limitation """
        # Get FFmpeg format string
        format_str = self._get_ffmpeg_format_string(segment)

        # Get audio parameters
        sample_rate = segment.frame_rate
        channels = segment.channels
        total_data_size = len(segment._data)  # pylint: disable=protected-access

        self.logger.info("Exporting to MP3 using FFmpeg (direct from PCM)")
        self.logger.debug(f"Format: {format_str}, {sample_rate}Hz, {channels}ch, {bitrate}bps")

        # Build FFmpeg command
        cmd = [
            'ffmpeg',
            '-f', format_str,  # Input format
            '-ar', str(sample_rate),  # Sample rate
            '-ac', str(channels),  # Channels
            '-i', 'pipe:0',  # Read from stdin
            '-codec:a', 'libmp3lame',  # MP3 codec
            '-b:a', bitrate,  # Bitrate
            '-y',  # Overwrite output file
            output_path
        ]

        try:
            # Start FFmpeg process
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE
            )

            # Send audio data in chunks to track progress
            chunk_size = 1024 * 1024  # 1MB chunks
            sent = 0
            last_logged_percent = -1

            while sent < total_data_size:
                chunk = segment._data[sent:sent + chunk_size]  # pylint: disable=protected-access
                process.stdin.write(chunk)
                sent += len(chunk)

                # Log progress every 5%
                percent = (sent / total_data_size) * 100
                if int(percent) // 5 > last_logged_percent // 5:
                    last_logged_percent = int(percent)
                    self.logger.debug(f"Encoding progress: {percent:.1f}%")

            process.stdin.close()

            # Wait for completion
            return_code = process.wait()

            if return_code != 0:
                stderr_output = process.stderr.read().decode('utf-8', errors='ignore')
                raise RuntimeError(f"FFmpeg failed with return code {return_code}\n{stderr_output}")

            self.logger.debug("Encoding complete: 100.0%")

        except FileNotFoundError as exc:
            raise RuntimeError(
                "FFmpeg not found. Please install FFmpeg (brew install ffmpeg or apt-get install ffmpeg)"
            ) from exc

    def execute_step(self, step_name: str, config_data: Any) -> None:
        """ Execute a single audio stitching step from the configuration """
        start_time = time.time()

        self.logger.header(f"Executing step: {step_name}")

        # Get step configuration
        step_config = config_data.steps.get(step_name)
        if not step_config:
            raise ValueError(f"Step '{step_name}' not found in configuration")

        # Validate that this is an AudioStitcherStep
        if not hasattr(step_config, 'stitch_fade') or not hasattr(step_config, 'normalisation'):
            raise ValueError(f"Step '{step_name}' is not configured for audio stitching")

        self.logger.info(f"Step: {step_name}")
        self.logger.info(f"Module: {step_config.module}")
        self.logger.info(f"Input: {step_config.input}")
        self.logger.info(f"Output: {step_config.output}")
        self.logger.info(f"Stitch fade duration: {step_config.stitch_fade}s")
        self.logger.info(f"Intro fade duration: {step_config.intro_fade}s")
        self.logger.info(f"Outro fade duration: {step_config.outro_fade}s")
        self.logger.info(f"Normalization headroom: {step_config.normalisation}dB")

        # Step 1: Identify samples
        self.logger.info("Step 1: Identifying audio samples")
        samples = self.identify_samples(step_config.input)

        # Step 2: Normalize samples
        self.logger.info("Step 2: Normalizing samples")
        normalized_samples = self.normalize_samples(samples, step_config.normalisation)

        # Step 3: Stitch samples
        self.logger.info("Step 3: Stitching samples")
        intermediate_output = f"{step_config.output.split('.')[0]}_intermediate.{step_config.output.split('.')[1]}"
        self.stitch_samples(normalized_samples, step_config.stitch_fade, intermediate_output)

        # Step 4: Loop to video length
        self.logger.info("Step 4: Looping to video length")
        intermediate_path = Paths.get_interim_path(intermediate_output)
        self.loop_to_video_length(intermediate_path, config_data, step_config)

        # Step 5: Cleanup intermediate files
        self.logger.info("Step 5: Cleaning up intermediate files")
        self.cleanup_intermediate_files(normalized_samples, intermediate_output)

        # Log completion
        duration = time.time() - start_time
        self.logger.header("Step completed successfully")
        self.logger.info(f"Step name: {step_name}")
        self.logger.info(f"Module: {step_config.module}")
        self.logger.info(f"Input samples: {len(samples)}")
        self.logger.info(f"Output file: {step_config.output}")
        self.logger.info(f"Duration: {duration:.2f} seconds")


def main() -> None:
    """ Main entry point """
    # Parse arguments
    args = parse_arguments("Audio stitcher module for combining multiple audio samples")

    try:
        # Load configuration first to get debug setting
        config_path = Project.get_configuration()
        config_loader = load_configuration(config_path)
        config_data = config_loader.data

        # Setup logging with debug setting from config
        logger = EnhancedLogger.setup_pipeline_logging(
            log_file=args.log_file,
            debug=config_data.debug or False,
            script_name="audio_stitcher"
        )

        logger.info(f"Loaded configuration: {config_data.name}")

        # Initialize audio stitcher
        stitcher = AudioStitcher(logger)

        # Execute the step
        stitcher.execute_step(args.step, config_data)

    except Exception as exc:
        logger.error(f"Audio stitching failed: {exc}")
        raise


if __name__ == "__main__":
    main()
