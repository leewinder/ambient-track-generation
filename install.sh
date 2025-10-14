#!/bin/bash
set -euo pipefail  # Exit on any error, unset variables, and pipe failures

# Parse command line arguments
SKIP_ENVIRONMENTS=false
TARGET_MODULE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-environments)
            SKIP_ENVIRONMENTS=true
            shift
            ;;
        --module)
            TARGET_MODULE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--no-environments] [--module <module_or_utility_name>]"
            exit 1
            ;;
    esac
done

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
required_version=$(cat .python-version | tr -d '\n')
echo "Required Python version: $required_version"

# Check if version is already installed
installed_versions=$(pyenv versions --bare)
if echo "$installed_versions" | grep -Fxq "$required_version"; then
    echo "Python $required_version is already installed"
else
    echo "Installing Python $required_version..."
    pyenv install "$required_version"
fi

# Set local Python version
echo "Setting local Python version to $required_version..."
pyenv local "$required_version"

# Refresh shell to use new Python version
eval "$(pyenv init -)"

# Verify Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Current Python version: $python_version"

if [[ ! "$python_version" =~ ^3\.11\. ]]; then
    echo "Wrong Python version. Expected 3.11.x, got $python_version"
    exit 1
fi

# Only create virtual environments if -no-environments flag is not set
if [ "$SKIP_ENVIRONMENTS" = false ]; then
    echo ""
    echo "***** Create per-stage virtual environments *****"

    # Discover all module directories dynamically
    echo "Discovering modules in src/modules/..."
    module_folders=()
    for module_dir in src/modules/*/; do
        if [[ -d "$module_dir" && -f "${module_dir}requirements.txt" ]]; then
            module_folders+=("$module_dir")
            echo "Found module: $module_dir"
        fi
    done

    # Discover all utility directories dynamically
    echo "Discovering utilities in utilities/..."
    utility_folders=()
    for utility_dir in utilities/*/; do
        if [[ -d "$utility_dir" && -f "${utility_dir}requirements.txt" ]]; then
            utility_folders+=("$utility_dir")
            echo "Found utility: $utility_dir"
        fi
    done

    # Combine modules and utilities into a single array
    all_folders=("${module_folders[@]}" "${utility_folders[@]}")

    if [[ ${#all_folders[@]} -eq 0 ]]; then
        echo "No modules or utilities found with requirements.txt files"
        exit 1
    fi

    echo "Total folders to process: ${#all_folders[@]} (${#module_folders[@]} modules, ${#utility_folders[@]} utilities)"

    # Loop through discovered folders (modules and utilities)
    for folder in "${all_folders[@]}"; do
        # Extract folder name from path (e.g., "src/modules/content_generation/" -> "content_generation")
        folder_name=$(basename "$folder")
        
        # Skip folders if TARGET_MODULE is set and doesn't match
        if [[ -n "$TARGET_MODULE" && "$folder_name" != "$TARGET_MODULE" ]]; then
            echo "Skipping $folder_name (not matching target: $TARGET_MODULE)"
            continue
        fi

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

        # Install requirements
        echo "Installing requirements from requirements.txt..."
        pip install -r requirements.txt

        # Deactivate
        echo "Closing down virtual environment..."
        deactivate

        cd - >/dev/null
    done
else
    echo ""
    echo "***** Skipping virtual environment creation (--no-environments flag provided) *****"
fi


echo ""
echo ""
echo "All installations complete!"
echo ""
echo ""