#!/bin/bash

# PostgreSQL shutdown script for Singularity/Apptainer

POSTGRES_INSTANCE="${POSTGRES_INSTANCE:-postgres_instance}"
POSTGRES_PID_FILE="${POSTGRES_PID_FILE:-postgres-${POSTGRES_INSTANCE}.pid}"

# Detect which container runtime is available
if command -v apptainer &> /dev/null; then
    CONTAINER_CMD="apptainer"
elif command -v singularity &> /dev/null; then
    CONTAINER_CMD="singularity"
else
    echo "ERROR: Neither apptainer nor singularity found in PATH"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Stopping PostgreSQL...${NC}"

STOPPED=false

# Method 1: Stop instance if it exists
if $CONTAINER_CMD instance list | awk -v name="${POSTGRES_INSTANCE}" '$1 == name { found = 1 } END { exit !found }'; then
    echo -e "${YELLOW}Stopping ${CONTAINER_CMD} instance: ${POSTGRES_INSTANCE}${NC}"
    if ! $CONTAINER_CMD instance stop "${POSTGRES_INSTANCE}"; then
        $CONTAINER_CMD instance stop --force "${POSTGRES_INSTANCE}" || exit 1
    fi
    STOPPED=true
fi

if [ "$STOPPED" = true ]; then
    echo -e "${GREEN}PostgreSQL stopped${NC}"
else
    echo -e "${YELLOW}No PostgreSQL instances were found running${NC}"
fi
