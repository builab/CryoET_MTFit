#!/bin/bash
# Install the MTFit ChimeraX bundle with all dependencies

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHL="$SCRIPT_DIR/ChimeraX-MTFit/dist/chimerax_mtfit-1.0.0-py3-none-any.whl"

# Find ChimeraX
if command -v ChimeraX &>/dev/null; then
    CHIMERAX="ChimeraX"
elif [ -f "/Applications/ChimeraX-1.10.app/Contents/bin/ChimeraX" ]; then
    CHIMERAX="/Applications/ChimeraX-1.10.app/Contents/bin/ChimeraX"
else
    echo "Error: ChimeraX not found. Please install ChimeraX first."
    exit 1
fi

echo "Installing MTFit bundle..."
"$CHIMERAX" --nogui --cmd "toolshed uninstall ChimeraX-MTFit; exit" 2>/dev/null || true
"$CHIMERAX" --nogui --cmd "toolshed install $WHL; exit"
echo "Done. Open ChimeraX and run: mtfit #1"
