#!/bin/bash
set -e

PACKAGE=idps
VERSION=1.0.0
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$(mktemp -d)"

echo "Building $PACKAGE $VERSION RPM ..."

# Set up rpmbuild directory structure
mkdir -p "$BUILD_DIR"/{BUILD,RPMS,SOURCES,SPECS,SRPMS}

# Copy source files into SOURCES
cp "$SCRIPT_DIR/main.py"             "$BUILD_DIR/SOURCES/"
cp "$SCRIPT_DIR/model.py"            "$BUILD_DIR/SOURCES/"
cp "$SCRIPT_DIR/db.py"               "$BUILD_DIR/SOURCES/"
cp "$SCRIPT_DIR/config.py"           "$BUILD_DIR/SOURCES/"
cp "$SCRIPT_DIR/flows_to_tensors.py" "$BUILD_DIR/SOURCES/"
cp "$SCRIPT_DIR/sniff.py"            "$BUILD_DIR/SOURCES/"
cp "$SCRIPT_DIR/requirements.txt"    "$BUILD_DIR/SOURCES/"
mkdir -p "$BUILD_DIR/SOURCES/util"
cp "$SCRIPT_DIR/util/"*.py           "$BUILD_DIR/SOURCES/util/"
mkdir -p "$BUILD_DIR/SOURCES/models"
cp "$SCRIPT_DIR/models/idps.weights.h5"                "$BUILD_DIR/SOURCES/models/"
cp "$SCRIPT_DIR/models/idps.weights.h5.scaler.joblib"  "$BUILD_DIR/SOURCES/models/"
cp "$SCRIPT_DIR/idps.service"        "$BUILD_DIR/SOURCES/"

# Copy spec file
cp "$SCRIPT_DIR/rpm/idps.spec" "$BUILD_DIR/SPECS/"

# Build the RPM
rpmbuild --define "_topdir $BUILD_DIR" -bb "$BUILD_DIR/SPECS/idps.spec"

# Copy the built RPM to the script directory
cp "$BUILD_DIR/RPMS/noarch/"*.rpm "$SCRIPT_DIR/"

# Clean up
rm -rf "$BUILD_DIR"

echo "Built: $(ls "$SCRIPT_DIR"/*.rpm)"
