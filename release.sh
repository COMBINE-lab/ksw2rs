#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <version>"
    echo ""
    echo "Example: $0 0.2.0"
    echo ""
    echo "This script will:"
    echo "  1. Update the version in Cargo.toml"
    echo "  2. Commit the version bump"
    echo "  3. Create a git tag (v<version>)"
    echo "  4. Push the commit and tag"
    echo "  5. Publish to crates.io"
    exit 1
}

if [ $# -ne 1 ]; then
    usage
fi

VERSION="$1"
TAG="v${VERSION}"

# Validate version format (basic semver check)
if ! echo "$VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$'; then
    echo "Error: invalid semver version: $VERSION"
    exit 1
fi

# Ensure we're in the repo root
if [ ! -f Cargo.toml ]; then
    echo "Error: Cargo.toml not found. Run this from the repository root."
    exit 1
fi

# Ensure working tree is clean
if [ -n "$(git status --porcelain -- Cargo.toml Cargo.lock src/)" ]; then
    echo "Error: working tree has uncommitted changes in tracked files."
    echo "Please commit or stash them first."
    exit 1
fi

# Ensure tag doesn't already exist
if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "Error: tag $TAG already exists."
    exit 1
fi

# Read current version
CURRENT=$(grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)"/\1/')
echo "Current version: $CURRENT"
echo "New version:     $VERSION"
echo "Tag:             $TAG"
echo ""

read -rp "Proceed? [y/N] " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Update version in Cargo.toml
sed -i.bak "s/^version = \"$CURRENT\"/version = \"$VERSION\"/" Cargo.toml
rm -f Cargo.toml.bak

# Verify the build passes
echo "Running cargo check..."
cargo check --quiet

echo "Committing version bump..."
git add Cargo.toml
git commit -m "release: v${VERSION}"

echo "Tagging ${TAG}..."
git tag -a "$TAG" -m "Release ${VERSION}"

echo "Pushing commit and tag..."
git push
git push origin "$TAG"

echo "Publishing to crates.io..."
cargo publish

echo ""
echo "Released ${TAG} successfully."
