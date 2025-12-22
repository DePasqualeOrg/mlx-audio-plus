#!/bin/bash
set -e

# Change to project root directory
cd "$(dirname "$0")/.."

# Ensure build dependencies are installed
uv pip install --quiet --python .venv/bin/python build twine

# Get current version
CURRENT_VERSION=$(grep -o '"[0-9]*\.[0-9]*\.[0-9]*"' mlx_audio/version.py | tr -d '"')
echo "Current version: $CURRENT_VERSION"

# Parse version components
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"

# Bump patch version
NEW_PATCH=$((PATCH + 1))
NEW_VERSION="$MAJOR.$MINOR.$NEW_PATCH"
echo "New version: $NEW_VERSION"

# Confirmation prompt
read -p "Publish v$NEW_VERSION to PyPI? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted"
    exit 1
fi

# Update version.py
sed -i '' "s/__version__ = \"$CURRENT_VERSION\"/__version__ = \"$NEW_VERSION\"/" mlx_audio/version.py
echo "Updated mlx_audio/version.py"

# Clean and build
rm -rf dist/ build/ *.egg-info
.venv/bin/python -m build
echo "Build complete"

# Upload to PyPI
.venv/bin/python -m twine upload dist/*
echo "Published v$NEW_VERSION to PyPI"

# Commit and tag
git add mlx_audio/version.py
git commit -m "Publish v$NEW_VERSION"
git tag "v$NEW_VERSION"
echo "Created commit and tag v$NEW_VERSION"

# Push commit and tag
git push && git push --tags
echo "Pushed to remote"
