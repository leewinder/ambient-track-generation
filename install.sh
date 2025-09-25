#!/bin/bash
set -e  # Exit on any error

echo ""
echo ""
echo "==================================================="
echo "= Setting up ambient track generation environment ="
echo "==================================================="

echo ""
echo "***** Verifying and setting up Python environment *****"

# Check if pyenv is installed (REQUIRED)
if ! command -v pyenv &> /dev/null; then
    echo "ERROR: pyenv is required but not found!"
    echo "   Install pyenv from: https://github.com/pyenv/pyenv"
    echo "   This script cannot continue without pyenv."
    exit 1
fi

echo "pyenv detected"

# Check if the required Python version is installed
required_version=$(cat .python-version)
echo "Required Python version: $required_version"

if pyenv versions --bare | grep -q "^$required_version$"; then
    echo "Python $required_version is already installed"
else
    echo "Installing Python $required_version..."
    pyenv install $required_version
fi

# Set local Python version
echo "Setting local Python version to $required_version..."
pyenv local $required_version

# Refresh shell to use new Python version
eval "$(pyenv init -)"

# Verify Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Current Python version: $python_version"

if [[ ! "$python_version" =~ ^3\.11\. ]]; then
    echo "Wrong Python version. Expected 3.11.x, got $python_version"
    exit 1
fi

echo ""
echo "***** Create per-stage virtual environments *****"

# Optional: folder to process (first argument)
TARGET_FOLDER="$1"

# Define folders
folders=(
    "src" 
    "src/generation/01 - Generate Image" 
    "src/generation/02 - Expand Image"
    "src/generation/03 - Upscale Image"
)

# Define requirements for each folder (space-separated strings)
requirements_list=(
    "requirements/requirements.txt"
    "requirements/requirements-torch.txt requirements/requirements-all.txt"
    "requirements/requirements-all.txt"
    "requirements/requirements-all.txt"
)

# Loop through folders
for i in "${!folders[@]}"; do
    folder="${folders[$i]}"

    # Skip folders if TARGET_FOLDER is set and doesn't match
    if [[ -n "$TARGET_FOLDER" && "$folder" != "$TARGET_FOLDER" ]]; then
        continue
    fi

    reqs="${requirements_list[$i]}"

    echo ""
    echo "Setting up $folder..."
    cd "$folder"

    # Delete existing virtual environment if it exists
    if [ -d "venv" ]; then
        echo "Removing existing virtual environment..."
        rm -rf venv
    fi

    # Create and activate virtual environment
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate

    # Upgrade pip
    echo "Upgrading pip..."
    pip install --upgrade pip

    # Install all requirements
    for req_file in $reqs; do
        echo "Installing requirements from $req_file..."
        pip install -r "$req_file"
    done

    # Deactivate
    echo "Closing down virtual environment..."
    deactivate

    cd - >/dev/null
done


echo ""
echo "***** Downloading model weights *****"

# Create weights directory
WEIGHTS_DIR="models"
mkdir -p "$WEIGHTS_DIR"

# Array of weight downloads (URL:filename pairs)
declare -a weight_downloads=(
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth:RealESRGAN_x2plus.pth"
    # Add more weight downloads here in the future
    # "https://example.com/model2.pth:model2.pth"
)

# Track failed downloads
declare -a failed_downloads=()

# Function to download a weight file
download_weight() {
    local url="$1"
    local filename="$2"
    local filepath="$WEIGHTS_DIR/$filename"
    
    echo "Downloading $filename..."
    
    # Check if file already exists
    if [ -f "$filepath" ]; then
        echo "  $filename already exists, skipping download"
        return 0
    fi
    
    # Download with curl, follow redirects, and show progress
    if curl -L --fail --progress-bar -o "$filepath" "$url"; then
        echo "  ✓ Successfully downloaded $filename"
        return 0
    else
        echo "  ✗ Failed to download $filename from $url"
        failed_downloads+=("$filename:$url")
        # Remove partial file if it exists
        [ -f "$filepath" ] && rm -f "$filepath"
        return 1
    fi
}

# Download all weights
for weight_info in "${weight_downloads[@]}"; do
    # Split on the last colon to handle URLs that contain colons
    url="${weight_info%:*}"
    filename="${weight_info##*:}"
    download_weight "$url" "$filename"
done

echo ""
echo "***** Weight download summary *****"

if [ ${#failed_downloads[@]} -eq 0 ]; then
    echo "All weight files downloaded successfully!"
else
    echo "Some weight downloads failed:"
    for failed in "${failed_downloads[@]}"; do
        # Split on the last colon to handle URLs that contain colons
        filename="${failed%:*}"
        url="${failed##*:}"
        echo "  - $filename (from $url)"
    done
    echo ""
    echo "You may need to download these manually or check if the URLs are still valid."
fi

echo ""
echo ""
echo "All installations complete!"
echo ""
echo ""