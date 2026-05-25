#!/usr/bin/env bash

# Get the directory where this script is located
MTFIT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Export variables
export MTFIT

# Add script directory to PATH if needed
export PATH="$MTFIT/scripts:$PATH"
