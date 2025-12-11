#!/bin/bash
set -e

# Change to project root directory
cd "$(dirname "$0")/.."

# Get current version
CURRENT_VERSION=$(grep -o '"[0-9]*\.[0-9]*\.[0-9]*"' mlx_audio/version.py | tr -d '"')
echo "Current version: $CURRENT_VERSION"

# Parse version components
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"

# Bump patch version
NEW_PATCH=$((PATCH + 1))
NEW_VERSION="$MAJOR.$MINOR.$NEW_PATCH"
echo "New version: $NEW_VERSION"

# Update version.py
sed -i '' "s/__version__ = \"$CURRENT_VERSION\"/__version__ = \"$NEW_VERSION\"/" mlx_audio/version.py
echo "Updated mlx_audio/version.py"

# Commit and tag
git add mlx_audio/version.py
git commit -m "Publish v$NEW_VERSION"
git tag "v$NEW_VERSION"
echo "Created commit and tag v$NEW_VERSION"

# Push commit and tag
git push && git push --tags
echo "Pushed to remote"

# Clean and build
rm -rf dist/ build/ *.egg-info
.venv/bin/python -m build
echo "Build complete"

# Upload to PyPI
.venv/bin/python -m twine upload dist/*
echo "Published v$NEW_VERSION to PyPI"
