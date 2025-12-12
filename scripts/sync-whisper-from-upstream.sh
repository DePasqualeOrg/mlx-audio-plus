#!/bin/bash
#
# Sync Whisper core files from official mlx-examples repository
#
# This script syncs the following files from ml-explore/mlx-examples:
#   - decoding.py
#   - tokenizer.py
#   - timing.py
#   - writers.py
#
# These files are pure ML utilities with no mlx-audio-plus customizations.
# The custom functionality lives in whisper.py and audio.py, which we don't sync.
#
# For more details on the sync strategy, see: docs/stt/whisper.md
#
# Usage:
#   ./scripts/sync-whisper-from-upstream.sh [--test]
#
# Options:
#   --test    Run tests after syncing
#

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WHISPER_DIR="$PROJECT_ROOT/mlx_audio/stt/models/whisper"
TMP_DIR="${TMP_DIR:-/tmp}"
UPSTREAM_REPO="https://github.com/ml-explore/mlx-examples.git"
UPSTREAM_DIR="$TMP_DIR/mlx-examples-sync"

# Parse arguments
RUN_TESTS=false

for arg in "$@"; do
    case $arg in
        --test)
            RUN_TESTS=true
            ;;
        --help)
            grep '^#' "$0" | tail -n +2 | cut -c 3-
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

echo "==================================================="
echo "Syncing Whisper files from upstream mlx-examples"
echo "==================================================="
echo

# Clone or update the upstream repository
if [ -d "$UPSTREAM_DIR" ]; then
    echo "📦 Updating existing upstream repo at $UPSTREAM_DIR..."
    cd "$UPSTREAM_DIR"
    git fetch origin main
    git reset --hard origin/main
else
    echo "📦 Cloning upstream repo to $UPSTREAM_DIR..."
    git clone --depth 1 --branch main "$UPSTREAM_REPO" "$UPSTREAM_DIR"
fi

echo "✓ Upstream repo ready"
echo

# Get the upstream commit hash
cd "$UPSTREAM_DIR"
UPSTREAM_COMMIT=$(git rev-parse --short HEAD)
UPSTREAM_DATE=$(git log -1 --format=%cd --date=short)
echo "📍 Upstream commit: $UPSTREAM_COMMIT ($UPSTREAM_DATE)"
echo

# Files to sync
FILES=(
    "decoding.py"
    "tokenizer.py"
    "timing.py"
    "writers.py"
)

# Show what will be synced
echo "📋 Files to sync:"
for file in "${FILES[@]}"; do
    echo "   - $file"
done
echo

# Backup current files and show diff
cd "$PROJECT_ROOT"
echo "🔍 Checking for changes..."
echo

CHANGES_FOUND=false
for file in "${FILES[@]}"; do
    if ! cmp -s "$UPSTREAM_DIR/whisper/mlx_whisper/$file" "$WHISPER_DIR/$file"; then
        CHANGES_FOUND=true
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Changes in $file:"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        diff -u "$WHISPER_DIR/$file" "$UPSTREAM_DIR/whisper/mlx_whisper/$file" || true
        echo
    fi
done

if [ "$CHANGES_FOUND" = false ]; then
    echo "✓ No changes detected - all files are already in sync!"
    exit 0
fi

# Ask for confirmation
echo
read -p "📝 Proceed with sync? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Sync cancelled"
    exit 1
fi

# Perform the sync
echo
echo "🔄 Syncing files..."
for file in "${FILES[@]}"; do
    cp "$UPSTREAM_DIR/whisper/mlx_whisper/$file" "$WHISPER_DIR/$file"
    echo "   ✓ $file"
done
echo

# Run tests if requested
if [ "$RUN_TESTS" = true ]; then
    echo "🧪 Running tests..."
    echo

    # Activate virtual environment if it exists
    if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
        source "$PROJECT_ROOT/.venv/bin/activate"
    fi

    # Run Whisper tests
    python -m pytest mlx_audio/stt/tests/test_models.py::TestWhisperModel -v

    echo
    echo "✓ Tests passed"
    echo
fi

# Show summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Sync complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
git diff --stat "$WHISPER_DIR"
echo

echo "📚 Next steps:"
if [ "$RUN_TESTS" = false ]; then
    echo "   1. Run tests: python -m pytest mlx_audio/stt/tests/test_models.py -v"
fi
echo "   2. Review changes: git diff"
echo "   3. Commit changes"
echo
