#!/bin/bash
set -e

PACKAGE=idps
VERSION=1.0.0
BUILD_DIR="$(mktemp -d)"
PKG_ROOT="$BUILD_DIR/${PACKAGE}_${VERSION}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Building $PACKAGE $VERSION ..."

# Create directory structure
mkdir -p "$PKG_ROOT/DEBIAN"
mkdir -p "$PKG_ROOT/opt/idps/util"
mkdir -p "$PKG_ROOT/var/idps/models"
mkdir -p "$PKG_ROOT/lib/systemd/system"

# Copy DEBIAN control files
cp "$SCRIPT_DIR/debian/control"   "$PKG_ROOT/DEBIAN/control"
cp "$SCRIPT_DIR/debian/postinst"  "$PKG_ROOT/DEBIAN/postinst"
cp "$SCRIPT_DIR/debian/prerm"     "$PKG_ROOT/DEBIAN/prerm"
cp "$SCRIPT_DIR/debian/postrm"    "$PKG_ROOT/DEBIAN/postrm"
chmod 755 "$PKG_ROOT/DEBIAN/postinst" "$PKG_ROOT/DEBIAN/prerm" "$PKG_ROOT/DEBIAN/postrm"

# Copy application source files
cp "$SCRIPT_DIR/main.py"             "$PKG_ROOT/opt/idps/"
cp "$SCRIPT_DIR/model.py"            "$PKG_ROOT/opt/idps/"
cp "$SCRIPT_DIR/db.py"               "$PKG_ROOT/opt/idps/"
cp "$SCRIPT_DIR/config.py"           "$PKG_ROOT/opt/idps/"
cp "$SCRIPT_DIR/flows_to_tensors.py" "$PKG_ROOT/opt/idps/"
cp "$SCRIPT_DIR/sniff.py"            "$PKG_ROOT/opt/idps/"
cp "$SCRIPT_DIR/requirements.txt"    "$PKG_ROOT/opt/idps/"

# Copy util modules
cp "$SCRIPT_DIR/util/"*.py "$PKG_ROOT/opt/idps/util/"

# Copy model and scaler
cp "$SCRIPT_DIR/models/idps.weights.h5"                "$PKG_ROOT/var/idps/models/"
cp "$SCRIPT_DIR/models/idps.weights.h5.scaler.joblib"  "$PKG_ROOT/var/idps/models/"

# Copy systemd service
cp "$SCRIPT_DIR/idps.service" "$PKG_ROOT/lib/systemd/system/"

# Build the .deb
dpkg-deb --build "$PKG_ROOT" "${SCRIPT_DIR}/${PACKAGE}_${VERSION}.deb"

# Clean up
rm -rf "$BUILD_DIR"

echo "Built: ${SCRIPT_DIR}/${PACKAGE}_${VERSION}.deb"
