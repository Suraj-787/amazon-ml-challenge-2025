#!/bin/bash

# Fix permissions for output directories

echo "Fixing permissions for output directories..."

# Fix ownership
sudo chown -R $(whoami):$(whoami) output/ 2>/dev/null || chown -R $(whoami):$(whoami) output/ 2>/dev/null

# Fix permissions
sudo chmod -R 755 output/ 2>/dev/null || chmod -R 755 output/ 2>/dev/null

# Create improved subdirectory
mkdir -p output/improved

# Test write access
if touch output/.test_write 2>/dev/null; then
    rm output/.test_write
    echo "✓ Permissions fixed successfully!"
    echo "✓ You can now run: ./scripts/setup-and-run.sh"
else
    echo "✗ Still cannot write to output/ directory"
    echo "Please run manually:"
    echo "  sudo chown -R \$(whoami):\$(whoami) output/"
    echo "  sudo chmod -R 755 output/"
    exit 1
fi
