#!/bin/sh
# Entrypoint script for Guaipeca container
# Ensures /data directories exist with correct permissions, then drops to non-root user

set -e

# Create needed directories if they don't exist
mkdir -p /data/config /data/corpora /data/index/converted /data/models

# Fix ownership (we run as root here, then drop down)
chown -R guaipeca:guaipeca /data

# Drop to non-root user and exec the main command
exec runuser -u guaipeca -- "$@"