#!/usr/bin/env python3
""" Content generation module for executing ComfyUI workflows """

import json
import time
import uuid
from pathlib import Path
from typing import Dict, Any

from pipeline_utilities.args import parse_arguments
from pipeline_utilities.configuration import load_configuration
from pipeline_utilities.logs import EnhancedLogger
from pipeline_utilities.paths import Paths, Project
from pipeline_utilities.comfyui import ComfyUIServer, ComfyUIWorkflow, ComfyUIOutput
from pipeline_utilities.authentication import load_authentication


class ContentGenerator:
    """ Main content generation orchestrator """

    def __init__(self, logger: EnhancedLogger, auth_data: Any = None):
        """ Initialize the content generator """
        self.logger = logger
        self.auth_data = auth_data

    def load_workflow_file(self, workflow_name: str) -> Dict[str, Any]:
        """ Load workflow JSON file """
        workflow_path = Paths.get_workflows_path(f"{workflow_name}.json")

        if not workflow_path.exists():
            raise FileNotFoundError(f"Workflow file not found: {workflow_path}")

        try:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)

            self.logger.info(f"Loaded workflow: {workflow_name}")
            return workflow_data

        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in workflow file: {exc}") from exc

    def _execute_single_pass(self, step_config: Any, workflow_name: str,
                             workflow_config: Any, config_data: Any, output_filename: str,
                             seed_offset: int = 0) -> None:
        """ Execute a single pass of workflow generation """
        # Load workflow file
        workflow_data = self.load_workflow_file(workflow_name)

        # Store original seed and apply offset
        original_seed = config_data.generation.seed
        modified_seed = original_seed + seed_offset
        config_data.generation.seed = modified_seed

        if seed_offset > 0:
            self.logger.debug(
                f"Using modified seed: {modified_seed} (original: {original_seed}, offset: +{seed_offset})")

        try:
            # Initialize ComfyUI utilities
            comfyui_config = config_data.comfyui
            client_id = str(uuid.uuid4())

            server = ComfyUIServer(
                str(comfyui_config.server),
                comfyui_config.check_interval,
                self.logger
            )

            workflow = ComfyUIWorkflow(
                str(comfyui_config.server),
                client_id,
                self.logger
            )

            output = ComfyUIOutput(
                str(comfyui_config.server),
                comfyui_config.output,
                self.logger
            )

            # Apply modifiers to workflow
            if workflow_config.modifiers:
                workflow_data, modifications = workflow.apply_modifiers(
                    workflow_data,
                    workflow_config.modifiers,
                    step_config,
                    config_data,
                    self.auth_data
                )

                # Log modifications made
                if modifications:
                    self.logger.info(f"Applied {len(modifications)} modifier(s) to workflow")
                    for mod in modifications:
                        if mod['input_field'] == 'value':
                            self.logger.debug(f"  {mod['node_name']}: {mod['new_value']}")
                        else:
                            self.logger.debug(f"  {mod['node_name']} ({mod['input_field']}): {mod['new_value']}")
                else:
                    self.logger.debug("No modifiers were applied to workflow")
            else:
                self.logger.debug("No modifiers defined for workflow")

            # Wait for server availability
            server.wait_until_available()

            # Submit workflow
            prompt_id = workflow.submit(workflow_data)

            # Monitor execution
            workflow.monitor(prompt_id)

            # Get outputs
            outputs = output.get_outputs(prompt_id)

            # Copy output file
            output.copy_output_file(outputs, output_filename)

        finally:
            # Restore original seed
            config_data.generation.seed = original_seed

    def execute_step(self, step_name: str, config_data: Any) -> None:
        """ Execute a single step from the configuration """
        start_time = time.time()

        self.logger.header(f"Executing step: {step_name}")

        # Get step configuration
        step_config = config_data.steps.get(step_name)
        if not step_config:
            raise ValueError(f"Step '{step_name}' not found in configuration")

        # Get workflow configuration
        workflow_name = step_config.workflow
        workflow_config = config_data.workflows.get(workflow_name)
        if not workflow_config:
            raise ValueError(f"Workflow '{workflow_name}' not found in configuration")

        self.logger.info(f"Step: {step_name}")
        self.logger.info(f"Module: {step_config.module}")
        self.logger.info(f"Workflow: {workflow_name}")
        self.logger.info(f"Output: {step_config.output}")

        # Check if passes field is present
        if step_config.passes is None:
            # Execute once with original output filename and seed
            self.logger.info("Executing single pass (no passes field)")
            self._execute_single_pass(step_config, workflow_name,
                                      workflow_config, config_data, step_config.output,
                                      seed_offset=0)
        else:
            # Execute multiple passes with modified filenames and incremented seeds
            passes = step_config.passes
            self.logger.info(f"Executing {passes} passes")

            for pass_num in range(1, passes + 1):
                self.logger.info(f"Executing pass {pass_num}/{passes}")

                # Modify output filename to include pass number
                output_path = Path(step_config.output)
                stem = output_path.stem
                suffix = output_path.suffix
                modified_filename = f"{stem}_pass_{pass_num:03d}{suffix}"

                self.logger.debug(f"Output filename: {modified_filename}")

                # Calculate seed offset (pass 1 = +0, pass 2 = +1, pass 3 = +2, etc.)
                seed_offset = pass_num - 1

                self._execute_single_pass(step_config, workflow_name,
                                          workflow_config, config_data, modified_filename,
                                          seed_offset=seed_offset)

        # Log completion
        duration = time.time() - start_time
        self.logger.header("Step completed successfully")
        self.logger.info(f"Step name: {step_name}")
        self.logger.info(f"Workflow: {workflow_name}")
        self.logger.info(f"Module: {step_config.module}")
        if step_config.passes is None:
            self.logger.info(f"Output file: {step_config.output}")
        else:
            self.logger.info(f"Output files: {step_config.passes} files with _pass_XXX suffix")
        self.logger.info(f"Duration: {duration:.2f} seconds")


def main() -> None:
    """ Main entry point """
    # Parse arguments
    args = parse_arguments("Content generation module for ComfyUI workflows")

    try:
        # Load configuration first to get debug setting
        config_path = Project.get_configuration()
        config_loader = load_configuration(config_path)
        config_data = config_loader.data

        # Setup logging with debug setting from config
        logger = EnhancedLogger.setup_pipeline_logging(
            log_file=args.log_file,
            debug=config_data.debug or False,
            script_name="content_generation"
        )

        logger.info(f"Loaded configuration: {config_data.name}")

        # Load authentication data (if available)
        auth_data = None
        try:
            auth_path = Project.get_root_path("authentication.json")
            auth_loader = load_authentication(auth_path)
            auth_data = auth_loader.data
            logger.info("Loaded authentication data")
        except FileNotFoundError:
            logger.info("No authentication.json found - authentication placeholders will not be available")
        except ValueError as exc:
            logger.warning(f"Failed to load authentication data: {exc}")

        # Initialize content generator
        generator = ContentGenerator(logger, auth_data)

        # Execute the step
        generator.execute_step(args.step, config_data)

    except Exception as exc:
        logger.error(f"Content generation failed: {exc}")
        raise


if __name__ == "__main__":
    main()
