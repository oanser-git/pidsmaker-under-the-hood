#!/bin/bash

POSTGRES_INSTANCE="${POSTGRES_INSTANCE:-postgres_instance}"

if command -v apptainer &> /dev/null; then
    CONTAINER_CMD="apptainer"
elif command -v singularity &> /dev/null; then
    CONTAINER_CMD="singularity"
else
    echo "ERROR: Neither apptainer nor singularity found in PATH"
    exit 1
fi

if $CONTAINER_CMD instance list | awk -v name="${POSTGRES_INSTANCE}" '$1 == name { found = 1 } END { exit !found }'; then
    if ! $CONTAINER_CMD instance stop "${POSTGRES_INSTANCE}"; then
        $CONTAINER_CMD instance stop --force "${POSTGRES_INSTANCE}" || exit 1
    fi
fi
