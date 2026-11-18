#!/usr/bin/env bash
set -e

# ! For versions like :latest, manually refresh once a more updated version is pushed to the registry

IMAGE_REF=$1
# Default cache: <data folder>/docker_cache; GENTIC_DATA_DIR overrides the data folder (unset = the submitted runs' one)
CACHE_DIR=${2:-${GENTIC_DATA_DIR:-/hpelustre/krai/krai_gentic}/docker_cache}

mkdir -p "$CACHE_DIR"

IMAGE_KEY=$(printf '%s' "$IMAGE_REF" | tr '/:@' '___')
ARCHIVE_PATH="${CACHE_DIR}/${IMAGE_KEY}.tar"

if docker image inspect "$IMAGE_REF" > /dev/null 2>&1; then
    echo "Image already present locally, skipping load"
    exit 0
fi

if [ -f "$ARCHIVE_PATH" ]; then
    echo "Found local cache, loading..."
    docker image load --input "$ARCHIVE_PATH"
else
    echo "Unable to locate local cached image, pulling..."
    docker pull "$IMAGE_REF"
    echo "Saving image to $ARCHIVE_PATH"
    TMP_PATH=$(mktemp "$ARCHIVE_PATH.tmp.XXXXXX")
    trap 'rm -f "$TMP_PATH"' EXIT
    docker image save --output "$TMP_PATH" "$IMAGE_REF"
    mv "$TMP_PATH" "$ARCHIVE_PATH"
fi