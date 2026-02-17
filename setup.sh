#!/bin/bash
# tasks/setup.sh - SWE-bench environment setup script
# Usage: ./setup.sh <base_commit>

set -euo pipefail

BASE_COMMIT="${1:-HEAD}"

# 1. Set locale environment variables
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# 2. Navigate to repository
cd /testbed

# 3. Export ROOT variable
export ROOT=$(pwd -P)

# 4. Reset git repository to base commit
git status
git restore .
git reset --hard "$BASE_COMMIT"
git clean -fdq