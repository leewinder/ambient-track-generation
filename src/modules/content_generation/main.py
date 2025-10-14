#!/usr/bin/env python3
""" Content generation module for executing ComfyUI workflows """

import json
import time
import uuid
from typing import Dict, Any

from pipeline_utilities.args import parse_arguments
from pipeline_utilities.configuration import load_configuration
from pipeline_utilities.logs import EnhancedLogger
from pipeline_utilities.paths import Paths, Project
from pipeline_utilities.comfyui import ComfyUIServer, ComfyUIWorkflow, ComfyUIOutput


class ContentGenerator:
    """ Main content generation orchestrator """

    def __init__(self, logger: EnhancedLogger):
        """ Initialize the content generator """
        self.logger = logger

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

        # Load workflow file
        workflow_data = self.load_workflow_file(workflow_name)

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
                config_data
            )

            # Log modifications made
            if modifications:
                self.logger.info("Applied modifiers to workflow:")
                for mod in modifications:
                    if mod['input_field'] == 'value':
                        self.logger.info(f"  {mod['node_name']}: {mod['new_value']}")
                    else:
                        self.logger.info(f"  {mod['node_name']} ({mod['input_field']}): {mod['new_value']}")
            else:
                self.logger.info("No modifiers were applied to workflow")
        else:
            self.logger.info("No modifiers defined for workflow")

        # Wait for server availability
        server.wait_until_available()

        # Submit workflow
        prompt_id = workflow.submit(workflow_data)

        # Monitor execution
        workflow.monitor(prompt_id)

        # Get outputs
        outputs = output.get_outputs(prompt_id)

        # Copy output file
        output.copy_output_file(outputs, step_config.output)

        # Log completion
        duration = time.time() - start_time
        self.logger.header("Step completed successfully")
        self.logger.info(f"Step name: {step_name}")
        self.logger.info(f"Workflow: {workflow_name}")
        self.logger.info(f"Module: {step_config.module}")
        self.logger.info(f"Output file: {step_config.output}")
        self.logger.info(f"Duration: {duration:.2f} seconds")


def main() -> None:
    """ Main entry point """
    # Parse arguments
    args = parse_arguments("Content generation module for ComfyUI workflows")

    # Setup logging
    logger = EnhancedLogger.setup_pipeline_logging(
        log_file=args.log_file,
        debug=False,  # Could be made configurable
        script_name="content_generation"
    )

    try:
        # Load configuration
        config_path = Project.get_configuration()
        config_loader = load_configuration(config_path)
        config_data = config_loader.data

        logger.info(f"Loaded configuration: {config_data.name}")

        # Initialize content generator
        generator = ContentGenerator(logger)

        # Execute the step
        generator.execute_step(args.step, config_data)

    except Exception as exc:
        logger.error(f"Content generation failed: {exc}")
        raise


if __name__ == "__main__":
    main()
