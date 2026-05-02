#!/usr/bin/env bash
# SPDX-License-Identifier: MIT

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
SANDBOX_DIR="${REPO_ROOT}/sandbox"
PYTHON_BIN=${PYTHON_BIN:-python3}

if [[ ! -d "${SANDBOX_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${SANDBOX_DIR}"
fi

"${SANDBOX_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${SANDBOX_DIR}/bin/python" -m pip install numpy torch --index-url https://download.pytorch.org/whl/cpu

cat <<EOF
Sandbox ready at ${SANDBOX_DIR}

Activate with:
  source "${SANDBOX_DIR}/bin/activate"
EOF
