#!/bin/bash

# Clear Hugging Face Cache Lock Files
# This script removes all lock files from the Hugging Face cache directory
# to prevent lock acquisition errors when downloading models

set -e  # Exit on any error

echo "🔓 Clearing Hugging Face cache lock files..."

# Check if the cache directory exists
CACHE_DIR="$HOME/.cache/huggingface/hub"
if [ ! -d "$CACHE_DIR" ]; then
    echo "❌ Hugging Face cache directory not found: $CACHE_DIR"
    echo "   This is normal if you haven't downloaded any models yet."
    exit 0
fi

# Count existing lock files
LOCK_COUNT=$(find "$CACHE_DIR" -name "*.lock" -type f 2>/dev/null | wc -l)

if [ "$LOCK_COUNT" -eq 0 ]; then
    echo "✅ No lock files found - cache is already clean!"
    exit 0
fi

echo "📊 Found $LOCK_COUNT lock file(s) to remove"

# Remove all lock files
echo "🗑️  Removing lock files..."
find "$CACHE_DIR" -name "*.lock" -type f -delete 2>/dev/null

# Verify removal
REMAINING_LOCKS=$(find "$CACHE_DIR" -name "*.lock" -type f 2>/dev/null | wc -l)

if [ "$REMAINING_LOCKS" -eq 0 ]; then
    echo "✅ Successfully cleared all $LOCK_COUNT lock file(s)"
    echo "🎉 Hugging Face cache is now clean and ready to use!"
else
    echo "⚠️  Warning: $REMAINING_LOCKS lock file(s) could not be removed"
    echo "   This might be due to active downloads or permission issues"
fi

echo ""
echo "💡 Tip: Run this script whenever you encounter lock acquisition errors"
echo "   when downloading or accessing Hugging Face models"
