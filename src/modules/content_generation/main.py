#!/usr/bin/env python3
""" Content generation module for executing ComfyUI workflows """

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from urllib.parse import urljoin

import requests
import websocket

from pipeline_utilities.args import parse_arguments
from pipeline_utilities.configuration import load_configuration
from pipeline_utilities.logs import EnhancedLogger
from pipeline_utilities.paths import Paths, Project


class ComfyUIClient:
    """ Client for interacting with ComfyUI server """

    def __init__(self, base_url: str, check_interval: int, output_path: str, logger: EnhancedLogger):
        """ Initialize the ComfyUI client """
        self.base_url = base_url.rstrip('/')
        self.check_interval = check_interval
        self.output_path = output_path
        self.logger = logger
        self.client_id = str(uuid.uuid4())
        self.node_mapping = {}  # node_id -> node_info mapping
        self.progress_state = {
            'current_step': 0,
            'total_steps': 0,
            'current_node': None,
            'start_time': None
        }

    def check_server_available(self) -> bool:
        """ Check if ComfyUI server is available and not busy """
        try:
            response = requests.get(f"{self.base_url}/queue", timeout=5)
            response.raise_for_status()

            queue_data = response.json()
            queue_pending = queue_data.get('queue_pending', [])
            queue_running = queue_data.get('queue_running', [])

            is_available = len(queue_pending) == 0 and len(queue_running) == 0

            if not is_available:
                self.logger.info(f"Server busy - pending: {len(queue_pending)}, running: {len(queue_running)}")

            return is_available

        except requests.RequestException as exc:
            self.logger.error(f"Failed to check server availability: {exc}")
            return False

    def wait_for_server(self) -> None:
        """ Wait for server to become available """
        self.logger.info("Waiting for ComfyUI server to become available...")

        while not self.check_server_available():
            self.logger.info(f"Server busy, checking again in {self.check_interval} seconds...")
            time.sleep(self.check_interval)

        self.logger.info("Server is now available")

    def _create_node_mapping(self, workflow_data: Dict[str, Any]) -> None:
        """ Create mapping from node IDs to node information """
        self.node_mapping = {}

        for node_id, node_info in workflow_data.items():
            if isinstance(node_info, dict):
                class_type = node_info.get('class_type', 'Unknown')
                title = node_info.get('_meta', {}).get('title', class_type)

                self.node_mapping[node_id] = {
                    'class_type': class_type,
                    'title': title,
                    'inputs': node_info.get('inputs', {})
                }

        self.logger.info(f"Created mapping for {len(self.node_mapping)} nodes")

    def _get_node_display_name(self, node_id: str) -> str:
        """ Get human-readable name for a node """
        if node_id in self.node_mapping:
            node_info = self.node_mapping[node_id]
            return f"{node_info['title']} ({node_info['class_type']})"
        return f"Node {node_id}"

    def _calculate_progress_percentage(self) -> float:
        """ Calculate progress percentage """
        if self.progress_state['total_steps'] > 0:
            return (self.progress_state['current_step'] / self.progress_state['total_steps']) * 100
        return 0.0

    def _estimate_remaining_time(self) -> str:
        """ Estimate remaining time based on current progress """
        if self.progress_state['start_time'] and self.progress_state['current_step'] > 0:
            elapsed = time.time() - self.progress_state['start_time']
            if self.progress_state['total_steps'] > 0:
                avg_time_per_step = elapsed / self.progress_state['current_step']
                remaining_steps = self.progress_state['total_steps'] - self.progress_state['current_step']
                remaining_time = avg_time_per_step * remaining_steps

                if remaining_time < 60:
                    return f"~{remaining_time:.0f}s"
                elif remaining_time < 3600:
                    return f"~{remaining_time/60:.0f}m"
                else:
                    return f"~{remaining_time/3600:.1f}h"
        return ""

    def submit_workflow(self, workflow_data: Dict[str, Any]) -> str:
        """ Submit workflow to ComfyUI and return prompt_id """
        try:
            # Create node mapping for progress tracking
            self._create_node_mapping(workflow_data)

            payload = {
                "prompt": workflow_data,
                "client_id": self.client_id
            }

            response = requests.post(
                f"{self.base_url}/prompt",
                json=payload,
                timeout=30
            )

            # Log the response details for debugging
            self.logger.info(f"ComfyUI response status: {response.status_code}")
            if response.status_code != 200:
                self.logger.error(f"ComfyUI error response: {response.text}")

            response.raise_for_status()

            result = response.json()
            prompt_id = result.get('prompt_id')

            if not prompt_id:
                raise RuntimeError("No prompt_id returned from ComfyUI")

            self.logger.info(f"Workflow submitted with prompt_id: {prompt_id}")
            return prompt_id

        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to submit workflow: {exc}") from exc

    def monitor_job(self, prompt_id: str) -> None:
        """ Monitor job execution via WebSocket with detailed progress tracking """
        self.logger.info(f"Starting detailed monitoring for job {prompt_id}...")

        ws_url = self.base_url.replace('http://', 'ws://').replace('https://', 'wss://')
        ws_url = f"{ws_url}/ws?clientId={self.client_id}"

        try:
            ws = websocket.WebSocket()
            ws.connect(ws_url, timeout=10)

            start_time = time.time()
            self.progress_state['start_time'] = start_time

            while True:
                # Set a shorter timeout for WebSocket messages to allow periodic checks
                ws.settimeout(5)

                try:
                    message = ws.recv()
                    data = json.loads(message)
                    message_type = data.get('type')

                    if message_type == 'execution_cached':
                        self.logger.info("Job using cached result")
                        # Don't break - continue monitoring for actual completion

                    elif message_type == 'execution_start':
                        self.logger.info("  Job execution started")

                    elif message_type == 'progress':
                        # Handle progress updates
                        progress_data = data.get('data', {})
                        current_step = progress_data.get('value', 0)
                        total_steps = progress_data.get('max', 0)
                        node_id = progress_data.get('node')

                        self.progress_state['current_step'] = current_step
                        self.progress_state['total_steps'] = total_steps
                        self.progress_state['current_node'] = node_id

                        if total_steps > 0:
                            percentage = self._calculate_progress_percentage()
                            remaining_time = self._estimate_remaining_time()
                            node_name = self._get_node_display_name(node_id) if node_id else "Unknown"

                            progress_msg = f"  Step {current_step}/{total_steps} ({percentage:.1f}%)"
                            if remaining_time:
                                progress_msg += f" - ETA: {remaining_time}"

                            self.logger.info(progress_msg)
                            if node_id:
                                self.logger.info(f"    Processing: {node_name}")

                    elif message_type == 'executing':
                        # Handle node execution start
                        node_id = data.get('data', {}).get('node')
                        if node_id and node_id != self.progress_state.get('current_node'):
                            node_name = self._get_node_display_name(node_id)
                            self.logger.info(f"  Starting: {node_name}")
                            self.progress_state['current_node'] = node_id

                    elif message_type == 'executed':
                        # Handle node completion
                        node_id = data.get('data', {}).get('node')
                        if node_id:
                            node_name = self._get_node_display_name(node_id)
                            self.logger.info(f"  Completed: {node_name}")

                            # Check if this is a SaveImage/SaveVideo/SaveAudio node (final output)
                            if node_id in self.node_mapping:
                                node_info = self.node_mapping[node_id]
                                class_type = node_info.get('class_type', '')
                                if class_type in ['SaveImage', 'SaveVideo', 'SaveAudio']:
                                    self.logger.info("  Final output node completed - checking job status")
                                    # Give a moment for the job to fully complete
                                    time.sleep(1)
                                    # Check if job is actually complete by querying history
                                    try:
                                        response = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=5)
                                        if response.status_code == 200:
                                            history_data = response.json()
                                            if prompt_id in history_data:
                                                self.logger.info("  Job execution completed successfully")
                                                break
                                    except requests.RequestException:
                                        # If we can't check history, assume it's complete
                                        self.logger.info("  Job execution completed successfully")
                                        break

                    elif message_type == 'execution_complete':
                        self.logger.info("  Job execution completed successfully")
                        break

                    elif message_type == 'execution_error':
                        error_data = data.get('data', {})
                        error_msg = error_data.get('exception_message', 'Unknown error')
                        node_id = error_data.get('node')

                        if node_id:
                            node_name = self._get_node_display_name(node_id)
                            raise RuntimeError(f"Job execution failed in {node_name}: {error_msg}")
                        else:
                            raise RuntimeError(f"Job execution failed: {error_msg}")

                except websocket.WebSocketTimeoutException:
                    # Periodic check: if we haven't received messages for a while,
                    # check if the job is actually complete
                    if self.progress_state.get('current_step', 0) > 0:
                        try:
                            response = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=5)
                            if response.status_code == 200:
                                history_data = response.json()
                                if prompt_id in history_data:
                                    self.logger.info(
                                        "  Job execution completed successfully (detected via periodic check)")
                                    break
                        except requests.RequestException:
                            pass
                    continue

        except websocket.WebSocketException as exc:
            raise RuntimeError(f"WebSocket error: {exc}") from exc
        finally:
            try:
                ws.close()
            except:
                pass

    def get_job_outputs(self, prompt_id: str) -> Dict[str, Any]:
        """ Get job outputs from ComfyUI history with retry logic """
        max_retries = 5
        retry_delay = 1.0  # Start with 1 second delay

        for attempt in range(max_retries):
            try:
                response = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=10)
                response.raise_for_status()

                history_data = response.json()
                prompt_data = history_data.get(prompt_id, {})

                if prompt_data:
                    outputs = prompt_data.get('outputs', {})

                    if outputs:
                        self.logger.info(f"Retrieved {len(outputs)} output nodes")
                        return outputs
                    else:
                        self.logger.warning(f"No outputs found for prompt_id: {prompt_id} (attempt {attempt + 1})")
                else:
                    self.logger.warning(f"No history found for prompt_id: {prompt_id} (attempt {attempt + 1})")

                # If we get here, history isn't ready yet - wait and retry
                if attempt < max_retries - 1:
                    self.logger.info(f"History not ready, retrying in {retry_delay:.1f}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 1.5  # Exponential backoff

            except requests.RequestException as exc:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Request failed (attempt {attempt + 1}): {exc}")
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
                else:
                    raise RuntimeError(f"Failed to get job outputs after {max_retries} attempts: {exc}") from exc

        # If we get here, all retries failed
        raise RuntimeError(f"No history found for prompt_id: {prompt_id} after {max_retries} attempts")

    def _get_comfyui_output_directory(self) -> Path:
        """ Get ComfyUI's output directory from configuration """
        # Expand user home directory if path starts with ~
        expanded_path = Path(self.output_path).expanduser()
        return expanded_path

    def copy_output_file(self, outputs: Dict[str, Any], output_filename: str) -> None:
        """ Copy output file from ComfyUI to interim directory """
        # Get ComfyUI's actual output directory
        comfyui_output_dir = self._get_comfyui_output_directory()

        # Find the output file in the outputs
        output_file_path = None

        for node_id, node_output in outputs.items():
            if 'images' in node_output:
                images = node_output['images']
                if images and len(images) > 0:
                    # Get the first image
                    image_info = images[0]
                    filename = image_info.get('filename', '')
                    subfolder = image_info.get('subfolder', '')

                    if filename:
                        # Construct the full path using ComfyUI's output directory
                        if subfolder:
                            full_output_dir = comfyui_output_dir / subfolder
                        else:
                            full_output_dir = comfyui_output_dir

                        output_file_path = full_output_dir / filename

                        if output_file_path.exists():
                            self.logger.info(f"Found output file: {output_file_path}")
                            break
                        else:
                            self.logger.warning(f"Expected file not found: {output_file_path}")

        if not output_file_path or not output_file_path.exists():
            # Log the full outputs structure for debugging
            self.logger.error(f"Output file not found. Full outputs structure:")
            self.logger.error(f"{json.dumps(outputs, indent=2)}")

            # List available files in the ComfyUI output directory
            self.logger.error(f"Available files in {comfyui_output_dir}:")
            if comfyui_output_dir.exists():
                files = list(comfyui_output_dir.rglob("*.png")) + \
                    list(comfyui_output_dir.rglob("*.jpg")) + list(comfyui_output_dir.rglob("*.jpeg"))
                self.logger.error(f"  {[f.name for f in files]}")
            else:
                self.logger.error(f"  Directory does not exist: {comfyui_output_dir}")

            raise FileNotFoundError(f"Output file not found: {output_file_path}")

        # Ensure interim directory exists
        interim_path = Paths.get_interim_path(output_filename)
        interim_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy the file
        shutil.copy2(output_file_path, interim_path)

        self.logger.info(f"Copied output to: {interim_path}")


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

    def apply_modifiers(self, workflow_data: Dict[str, Any], modifiers: Dict[str, Any],
                        step_config: Any, config_data: Any) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """ Apply modifiers to workflow data and return modified workflow with modifications list """
        if not modifiers:
            return workflow_data, []

        modifications = []
        errors = []

        # Create a mapping of node titles to node data for quick lookup
        node_title_map = {}
        # Handle new ComfyUI workflow format: flat dictionary with _meta.title
        if isinstance(workflow_data, dict):
            # Look through all top-level properties for nodes
            for node_id, node_data in workflow_data.items():
                if isinstance(node_data, dict) and '_meta' in node_data:
                    title = node_data['_meta'].get('title', '')
                    if title:
                        node_title_map[title] = node_data
        else:
            errors.append("Invalid workflow data format - expected dictionary")
            return workflow_data, []

        # Extract node titles that are being modified to check for duplicates only among those
        modifier_node_titles = set()
        for modifier_name in modifiers.keys():
            if '::' in modifier_name:
                node_title = modifier_name.split('::', 1)[0]
            else:
                node_title = modifier_name
            modifier_node_titles.add(node_title)

        # Check for duplicate titles only among nodes being modified
        # Count occurrences of each modifier node title in the workflow
        for title in modifier_node_titles:
            count = sum(1 for node_data in workflow_data.values()
                        if isinstance(node_data, dict) and
                        node_data.get('_meta', {}).get('title', '') == title)
            if count > 1:
                errors.append(f"Duplicate node title found: '{title}'")

        # Process each modifier
        for modifier_name, modifier_value in modifiers.items():
            # Resolve placeholder values
            resolved_value = self._resolve_placeholder(modifier_value, step_config, config_data)

            # Parse modifier name for input field specification
            if '::' in modifier_name:
                node_title, input_field = modifier_name.split('::', 1)
                if not node_title.strip():
                    errors.append(f"Modifier '{modifier_name}' has empty node title before '::'")
                    continue
                if not input_field.strip():
                    errors.append(f"Modifier '{modifier_name}' has empty input field after '::'")
                    continue
            else:
                node_title = modifier_name
                input_field = 'value'

            # Find the node by title
            if node_title not in node_title_map:
                errors.append(f"Node with title '{node_title}' not found in workflow")
                continue

            node = node_title_map[node_title]

            # Validate inputs field exists
            if 'inputs' not in node:
                errors.append(f"Node '{node_title}' does not have 'inputs' field")
                continue

            inputs = node['inputs']
            if not isinstance(inputs, dict):
                errors.append(f"Node '{node_title}' inputs field is not a dictionary")
                continue

            if input_field not in inputs:
                errors.append(f"Node '{node_title}' does not have 'inputs.{input_field}' field")
                continue

            # Validate type compatibility
            original_value = inputs[input_field]
            if not self._is_type_compatible(resolved_value, original_value):
                errors.append(
                    f"Type mismatch for node '{node_title}' field '{input_field}': "
                    f"modifier value type {type(resolved_value).__name__} "
                    f"does not match inputs.{input_field} type {type(original_value).__name__}"
                )
                continue

            # Apply the modifier
            inputs[input_field] = resolved_value
            modifications.append({
                'node_name': modifier_name,
                'node_title': node_title,
                'input_field': input_field,
                'old_value': original_value,
                'new_value': resolved_value
            })

        # If there are any errors, raise them all at once
        if errors:
            error_message = "Modifier validation errors:\n" + "\n".join(f"  - {error}" for error in errors)
            raise ValueError(error_message)

        return workflow_data, modifications

    def _resolve_placeholder(self, value: Any, step_config: Any, config_data: Any) -> Any:
        """ Resolve placeholder values like <prompts[0]>, <seed>, etc. """
        if not isinstance(value, str):
            return value

        # Handle <prompts[n]> placeholders
        if value.startswith('<prompts[') and value.endswith(']>'):
            try:
                # Extract index from <prompts[n]>
                index_str = value[9:-2]  # Remove '<prompts[' and ']>'
                prompt_index = int(index_str)

                # Validate index is within bounds
                if not step_config.prompts or prompt_index >= len(step_config.prompts):
                    raise ValueError(
                        f"Prompt index {prompt_index} is out of bounds (step has {len(step_config.prompts) if step_config.prompts else 0} prompts)")

                return step_config.prompts[prompt_index]

            except (ValueError, IndexError) as exc:
                raise ValueError(f"Invalid prompt placeholder '{value}': {exc}") from exc

        # Handle <seed> placeholder
        if value == '<seed>':
            return config_data.generation.seed

        # Handle <input> placeholder
        if value == '<input>':
            # Validate that input is configured
            if step_config.input is None:
                raise ValueError("Cannot use <input> placeholder: step.input is not configured")

            # Construct absolute path to interim file
            input_path = Paths.get_interim_path(step_config.input)

            # Validate file exists
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")

            # Return as absolute path string (ComfyUI expects string)
            return str(input_path)

        # Return as-is if not a recognized placeholder
        return value

    def _is_type_compatible(self, resolved_value: Any, original_value: Any) -> bool:
        """ Check if resolved value type is compatible with original value type """
        resolved_type = type(resolved_value)
        original_type = type(original_value)

        # Exact type match is always compatible
        if resolved_type == original_type:
            return True

        # Allow int/float compatibility for numeric values
        if isinstance(resolved_value, (int, float)) and isinstance(original_value, (int, float)):
            return True

        # All other type mismatches are incompatible
        return False

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

        # Apply modifiers to workflow
        if workflow_config.modifiers:
            workflow_data, modifications = self.apply_modifiers(
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

        # Initialize ComfyUI client
        comfyui_config = config_data.comfyui
        client = ComfyUIClient(
            str(comfyui_config.server),
            comfyui_config.check_interval,
            comfyui_config.output,
            self.logger
        )

        # Wait for server availability
        client.wait_for_server()

        # Submit workflow
        prompt_id = client.submit_workflow(workflow_data)

        # Monitor execution
        client.monitor_job(prompt_id)

        # Get outputs
        outputs = client.get_job_outputs(prompt_id)

        # Copy output file
        client.copy_output_file(outputs, step_config.output)

        # Log completion
        duration = time.time() - start_time
        self.logger.header(f"Step completed successfully")
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
