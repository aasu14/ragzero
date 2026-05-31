#!/usr/bin/env bash
# Build the ragzero wheel + sdist locally.
#
# Steps:
#   1. Build the React UI (produces ui/dist/)
#   2. Build the Python package (wheel + sdist, into dist/)
#
# Requires: Node.js, npm, Python with `build` package installed.

set -euo pipefail

cd "$(dirname "$0")"

echo "==> 1/2  Building UI"
if [ ! -d ui/node_modules ]; then
  echo "    Installing npm dependencies..."
  (cd ui && npm install)
fi
(cd ui && npm run build)

if [ ! -f ui/dist/index.html ]; then
  echo "ERROR: UI build did not produce ui/dist/index.html" >&2
  exit 1
fi
echo "    ui/dist/ produced ($(find ui/dist -type f | wc -l) files)"

echo
echo "==> 2/2  Building Python package"
if ! python3 -c "import build" >/dev/null 2>&1; then
  echo "    Installing the 'build' package..."
  python3 -m pip install --quiet build
fi

rm -rf dist/
python3 -m build

echo
echo "==> Done."
echo "    Artifacts in dist/:"
ls -lh dist/

echo
echo "Next steps:"
echo "  twine check dist/*           # validate the metadata"
echo "  twine upload --repository testpypi dist/*    # publish to TestPyPI first"
echo "  twine upload dist/*          # publish to real PyPI"
