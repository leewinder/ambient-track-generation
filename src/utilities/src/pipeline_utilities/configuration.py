#!/usr/bin/env python3
""" Pydantic models for configuration validation """

import time
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, HttpUrl, field_validator, model_validator

from .pydantics import StrictBaseModel, JsonFileLoader

# Public API - functions and classes that external scripts should use
__all__ = [
    'GenerationConfig',
    'VideoConfig',
    'ComfyUIConfig',
    'Workflow',
    'Step',
    'AudioStitcherStep',
    'VideoConstructionStep',
    'ArchiveVideoStep',
    'SunoAudioStep',
    'ConfigurationModel',
    'ConfigurationLoader',
    'load_configuration'
]


class GenerationConfig(StrictBaseModel):
    """ Configuration for generation settings """
    seed: Optional[int] = Field(
        default_factory=lambda: int(time.time()),
        description="Seed for generation, defaults to current epoch time",
        gt=0
    )


class VideoConfig(StrictBaseModel):
    """ Configuration for video settings """
    length: float = Field(
        description="Video length in minutes",
        gt=0
    )


class ComfyUIConfig(StrictBaseModel):
    """ Configuration for ComfyUI server settings """
    server: HttpUrl = Field(
        description="ComfyUI server address for API calls"
    )
    check_interval: Optional[int] = Field(
        default=10,
        description="Interval in seconds to check server availability",
        gt=0
    )
    output: str = Field(
        description="ComfyUI output directory path (required, no default)"
    )


class Workflow(StrictBaseModel):
    """ Workflow configuration """
    modifiers: Optional[Dict[str, Union[str, float, int]]] = Field(
        default_factory=dict,
        description="Dictionary of modifiers keyed by field name with string, float, or int values"
    )
    loras: Optional[Dict[str, float]] = Field(
        default_factory=dict,
        description="Dictionary of LoRAs keyed by name with weight values"
    )


class Step(StrictBaseModel):
    """ Step configuration """
    module: str = Field(
        description="Module name for the step",
        min_length=1
    )
    workflow: str = Field(
        description="Name of the workflow to use",
        min_length=1
    )
    input: Optional[str] = Field(
        default=None,
        description="Input for the step"
    )
    output: str = Field(
        description="Output for the step",
        min_length=1
    )
    prompts: Optional[List[str]] = Field(
        default_factory=list,
        description="Prompts for the step"
    )
    passes: Optional[int] = Field(
        default=None,
        description="Number of passes to execute (if present, outputs will be numbered)",
        gt=0
    )

    @field_validator('prompts')
    @classmethod
    def validate_prompts(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """ Ensure no empty strings in prompts list """
        if v is not None:
            for prompt in v:
                if not prompt or prompt.strip() == "":
                    raise ValueError("Prompts cannot contain empty strings")
        return v


class AudioStitcherStep(StrictBaseModel):
    """ Audio stitcher step configuration """
    module: str = Field(
        description="Module name for the step",
        min_length=1
    )
    input: Optional[str] = Field(
        default=None,
        description="Input for the step"
    )
    output: str = Field(
        description="Output for the step",
        min_length=1
    )
    stitch_fade: float = Field(
        description="Fade duration in seconds for cross-fading between samples",
        gt=0
    )
    intro_fade: float = Field(
        description="Fade-in duration in seconds for the final audio",
        ge=0
    )
    outro_fade: float = Field(
        description="Fade-out duration in seconds for the final audio",
        ge=0
    )
    normalisation: float = Field(
        description="Headroom in dB for audio normalization",
        ge=0
    )


class VideoConstructionStep(StrictBaseModel):
    """ Video construction step configuration """
    module: str = Field(
        description="Module name for the step",
        min_length=1
    )
    image: str = Field(
        description="Image filename for video construction",
        min_length=1
    )
    audio: str = Field(
        description="Audio filename for video construction",
        min_length=1
    )
    output: str = Field(
        description="Output video filename",
        min_length=1
    )


class ArchiveVideoStep(StrictBaseModel):
    """ Archive video step configuration """
    module: str = Field(
        description="Module name for the step",
        min_length=1
    )
    audio: str = Field(
        description="Regex pattern for matching audio files",
        min_length=1
    )
    artwork: str = Field(
        description="Regex pattern for matching artwork files",
        min_length=1
    )
    video: str = Field(
        description="Video filename to archive",
        min_length=1
    )


class SunoAudioStep(StrictBaseModel):
    """ Suno audio generation step configuration """
    module: str = Field(
        description="Module name for the step",
        min_length=1
    )
    output: str = Field(
        description="Output for the step",
        min_length=1
    )
    prompts: List[str] = Field(
        description="Prompts for the step",
        min_length=1
    )
    passes: Optional[int] = Field(
        default=None,
        description="Number of passes to execute (if present, outputs will be numbered)",
        gt=0
    )

    @field_validator('prompts')
    @classmethod
    def validate_prompts(cls, v: List[str]) -> List[str]:
        """ Ensure no empty strings in prompts list """
        for prompt in v:
            if not prompt or prompt.strip() == "":
                raise ValueError("Prompts cannot contain empty strings")
        return v


class ConfigurationModel(StrictBaseModel):
    """ Root configuration model """
    name: str = Field(
        description="Name of the configuration",
        min_length=1
    )
    debug: Optional[bool] = Field(
        default=False,
        description="Debug mode flag"
    )
    generation: Optional[GenerationConfig] = Field(
        default_factory=GenerationConfig,
        description="Generation configuration"
    )
    video: Optional[VideoConfig] = Field(
        default=None,
        description="Video configuration"
    )
    comfyui: Optional[ComfyUIConfig] = Field(
        default_factory=ComfyUIConfig,
        description="ComfyUI server configuration"
    )
    workflows: Dict[str, Workflow] = Field(
        description="Dictionary of workflows keyed by name",
        min_length=1
    )
    steps: Dict[str, Any] = Field(
        description="Dictionary of steps keyed by step name",
        min_length=1
    )

    @field_validator('steps')
    @classmethod
    def validate_steps(cls, steps: Dict[str, Any]) -> Dict[str, Any]:
        """ Validate and convert step data to appropriate step types """
        validated_steps = {}
        for step_name, step_data in steps.items():
            if isinstance(step_data, dict) and 'module' in step_data:
                module = step_data['module']
                if module == 'audio_stitcher':
                    # Validate AudioStitcherStep fields
                    required_fields = ['stitch_fade', 'intro_fade', 'outro_fade', 'normalisation']
                    for field in required_fields:
                        if field not in step_data:
                            raise ValueError(
                                f"Step '{step_name}' missing required field '{field}' for audio_stitcher module")
                    # Convert to AudioStitcherStep
                    validated_steps[step_name] = AudioStitcherStep(**step_data)
                elif module == 'video_construction':
                    # Validate VideoConstructionStep fields
                    required_fields = ['image', 'audio']
                    for field in required_fields:
                        if field not in step_data:
                            raise ValueError(
                                f"Step '{step_name}' missing required field '{field}' for video_construction module")
                    # Convert to VideoConstructionStep
                    validated_steps[step_name] = VideoConstructionStep(**step_data)
                elif module == 'archive_video':
                    # Validate ArchiveVideoStep fields
                    required_fields = ['audio', 'artwork', 'video']
                    for field in required_fields:
                        if field not in step_data:
                            raise ValueError(
                                f"Step '{step_name}' missing required field '{field}' for archive_video module")
                    # Convert to ArchiveVideoStep
                    validated_steps[step_name] = ArchiveVideoStep(**step_data)
                elif module == 'suno_audio':
                    # Validate SunoAudioStep fields
                    required_fields = ['output', 'prompts']
                    for field in required_fields:
                        if field not in step_data:
                            raise ValueError(
                                f"Step '{step_name}' missing required field '{field}' for suno_audio module")
                    # Convert to SunoAudioStep
                    validated_steps[step_name] = SunoAudioStep(**step_data)
                else:
                    # Regular Step - validate workflow reference
                    if 'workflow' in step_data:
                        # We'll validate workflow references in the model validator
                        pass
                    # Convert to Step
                    validated_steps[step_name] = Step(**step_data)
            else:
                validated_steps[step_name] = step_data
        return validated_steps

    @model_validator(mode='after')
    def validate_workflow_references(self) -> 'ConfigurationModel':
        """ Validate that step workflows reference existing workflow names """
        steps = getattr(self, 'steps', {})
        workflows = getattr(self, 'workflows', {})

        for step_name, step in steps.items():
            # Only validate workflow references for regular Step objects (skip special modules)
            if isinstance(step, Step) and hasattr(workflows, 'keys') and step.workflow not in list(workflows.keys()):
                raise ValueError(f"Step '{step_name}' references unknown workflow '{step.workflow}'")

        return self

    @model_validator(mode='after')
    def validate_prompt_counts(self) -> 'ConfigurationModel':
        """ Validate that step prompts count matches workflow modifiers prompt references """
        steps = getattr(self, 'steps', {})
        workflows = getattr(self, 'workflows', {})

        # Validate each step
        for step_name, step in steps.items():
            # Only validate prompt counts for regular Step objects
            if not isinstance(step, Step):
                continue

            workflow = workflows.get(step.workflow)
            if workflow and workflow.modifiers:
                # Find highest prompt index referenced in modifiers
                max_prompt_index = -1
                for modifier_value in workflow.modifiers.values():
                    if isinstance(modifier_value, str) and modifier_value.startswith('<prompts['):
                        # Extract index from <prompts[n]>
                        try:
                            end_bracket = modifier_value.find(']')
                            if end_bracket > 0:
                                index_str = modifier_value[8:end_bracket]  # Skip '<prompts['
                                prompt_index = int(index_str)
                                max_prompt_index = max(max_prompt_index, prompt_index)
                        except (ValueError, IndexError):
                            continue

                # Validate step has enough prompts
                if max_prompt_index >= 0:
                    step_prompts_count = len(step.prompts) if step.prompts else 0
                    required_count = max_prompt_index + 1

                    if step_prompts_count < required_count:
                        raise ValueError(
                            f"Step '{step_name}' has {step_prompts_count} prompts but workflow "
                            f"'{step.workflow}' modifiers reference prompts[0] through prompts[{max_prompt_index}] "
                            f"(requires {required_count} prompts)"
                        )

        return self


class ConfigurationLoader(JsonFileLoader):
    """ Loader for configuration files """

    def __init__(self, file_path: str):
        """ Initialize the configuration loader """
        super().__init__(file_path, ConfigurationModel)


# Convenience function for loading configuration files
load_configuration = JsonFileLoader.create_loader_function(ConfigurationLoader, "configuration.json")
