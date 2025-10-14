[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

# Ambient Track Generator

An AI-powered pipeline for (eventually) creating atmospheric videos by combining AI-generated images, upscaling them, adding subtle animations, and AI-generated music. The pipeline uses [ComfyUI](https://www.comfy.org/) for all AI generation tasks and orchestrates the entire workflow through a modular Python architecture.

## What It Does

So far, the project does the following...
- **Image Generation**: Creates initial atmospheric images using models via ComfyUI
- **Image Expansion**: Expands the generated images to 1080p images
- **Image Upscaling**: Upscales images using ESRGAN models to 4k and 8k

And eventually it will...
- **Animation**: Add subtle animation to the generated images to create short but loop'able videos
- **Music Generation**: Create multiple long-form ambient music tracks that blend together
- **Video Composition**: Bring the video and audio together to create multi-hour ambient music videos for platforms like YouTube

## Requirements

### System Requirements
- **MacOS with Apple Silicon**: There is no chance this works on anything else
- **Python**: 3.9+ (tested on 3.11.10)

### External Dependencies
- **ComfyUI**: Locally running instance accessible via HTTP API
- **AI Models**: Available via services like [HuggingFace](https://huggingface.co) or [Civitai](https://civitai.com)
- **pyenv**: For Python version management

### Python Dependencies
- **Core**: pydantic>=2.0.0
- **HTTP/WebSocket**: websocket-client>=1.6.0, requests>=2.31.0
- **Development**: autopep8, pylint

## Getting Started
### 1. Prerequisites

Install pyenv if you haven't already:
```bash
# macOS (using Homebrew)
brew install pyenv
# Add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
```

### 2. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/leewinder/ambient-track-generation.git
cd ambient-track-generation

# Run the automated setup script
./install.sh
```

The setup script will:
- Install Python 3.11.10 via pyenv
- Create virtual environments for each module
- Install all required dependencies
- Set up the project structure

### 3. Configure ComfyUI
Update `configuration.json` with your [ComfyUI](https://www.comfy.org/) server details:

```json
{
  "comfyui": {
    "server": "http://your-comfyui-server:8188/",
    "check_interval": 10,
    "output": "~/Documents/ComfyUI/output/"
  }
}
```

### 4. Run the Pipeline

```bash
# Run all pipeline steps
bash generate.sh
```

## Configuration

The `configuration.json` file controls the entire pipeline:

- **Generation Settings**: Seeds, debug mode
- **ComfyUI Connection**: Server URL, output paths
- **Workflow Definitions**: AI model parameters and prompts, with workflow overrides
- **Pipeline Steps**: Execution order and dependencies for each part of the pipeline

## Development

### Project Structure
```
ambient-track-generation/
├── configuration.json                # Main configuration file
├── install.sh                        # Automated setup script
├── .python-version                   # Python version specification
├── src/
│   ├── modules/                      # Pipeline execution modules
│   │   ├── content_generation/       # ComfyUI workflow execution
│   │   └── generation_runner/        # Pipeline orchestration
│   └── utilities/                    # Shared utilities library
│       └── src/pipeline_utilities/
├── workflows/                        # ComfyUI workflow definitions (exported via Export (API))
└── utilities/                        # Additional utility scripts
```


### Virtual Environments
Each module has its own virtual environment for isolation:
- `src/modules/content_generation/venv/`
- `src/modules/generation_runner/venv/`

### Adding New Modules
1. Create a new directory in `src/modules/`
2. Add `requirements.txt` and `main.py`
3. Run `./install.sh --module <module_name>` to set up the environment

## License
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.

## Contributing
1. Follow the established code style (PEP8)
2. Use type hints consistently
3. Document functions and classes clearly
4. Test changes thoroughly before submitting
