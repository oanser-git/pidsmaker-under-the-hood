#!/bin/bash

set -e

if command -v apptainer &> /dev/null; then
    CONTAINER_CMD="apptainer"
    export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${TMPDIR:-/tmp}/apptainer-${USER}}"
    export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${HOME}/.apptainer/cache}"
    export APPTAINER_SESSIONDIR="${APPTAINER_SESSIONDIR:-${TMPDIR:-/tmp}/apptainer-sessions-${USER}}"
    mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$APPTAINER_SESSIONDIR"
elif command -v singularity &> /dev/null; then
    CONTAINER_CMD="singularity"
    export SINGULARITY_TMPDIR="${TMPDIR:-/tmp}/singularity-${USER}"
    export SINGULARITY_CACHEDIR="${HOME}/.singularity/cache"
    mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"
else
    echo "ERROR: Neither apptainer nor singularity found in PATH"
    exit 1
fi

POSTGRES_IMAGE="${POSTGRES_IMAGE:-postgres.sif}"
POSTGRES_INSTANCE="${POSTGRES_INSTANCE:-postgres_instance}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
DATA_DIR="${POSTGRES_DATA_DIR:-postgres_data}"
RUN_DIR="${POSTGRES_RUN_DIR:-postgres_run}"
LOG_DIR="${POSTGRES_LOG_DIR:-postgres_log}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Starting PostgreSQL with ${CONTAINER_CMD}...${NC}"

if [ ! -f "$POSTGRES_IMAGE" ]; then
    echo -e "${YELLOW}PostgreSQL image not found. Pulling from Docker Hub...${NC}"
    $CONTAINER_CMD pull "$POSTGRES_IMAGE" docker://postgres:17
fi

mkdir -p "$DATA_DIR" "$RUN_DIR" "$LOG_DIR"

INPUT_DIR=${INPUT_DIR:-$(pwd)/data}
if [ ! -d "$INPUT_DIR" ]; then
    mkdir -p "$INPUT_DIR"
fi

if $CONTAINER_CMD instance list | awk -v name="$POSTGRES_INSTANCE" '$1 == name { found = 1 } END { exit !found }'; then
    if $CONTAINER_CMD exec "instance://$POSTGRES_INSTANCE" pg_isready -h localhost -p "$POSTGRES_PORT" -U postgres > /dev/null 2>&1; then
        exit 0
    fi
    $CONTAINER_CMD instance stop "$POSTGRES_INSTANCE"
    sleep 2
fi

if [ "$CONTAINER_CMD" = "apptainer" ]; then
    export APPTAINERENV_POSTGRES_PASSWORD=postgres
    export APPTAINERENV_POSTGRES_USER=postgres
    export APPTAINERENV_POSTGRES_DB=postgres
else
    export SINGULARITYENV_POSTGRES_PASSWORD=postgres
    export SINGULARITYENV_POSTGRES_USER=postgres
    export SINGULARITYENV_POSTGRES_DB=postgres
fi

BIND_MOUNTS="--bind $DATA_DIR:/var/lib/postgresql/data"
BIND_MOUNTS="$BIND_MOUNTS --bind $RUN_DIR:/var/run/postgresql"
BIND_MOUNTS="$BIND_MOUNTS --bind $LOG_DIR:/var/log"
BIND_MOUNTS="$BIND_MOUNTS --bind ./:/scripts"
BIND_MOUNTS="$BIND_MOUNTS --bind $INPUT_DIR:/data"

if [ -f "./postgres_config/postgresql.conf" ]; then
    BIND_MOUNTS="$BIND_MOUNTS --bind ./postgres_config/postgresql.conf:/etc/postgresql/postgresql.conf"
fi

$CONTAINER_CMD instance start $BIND_MOUNTS "$POSTGRES_IMAGE" "$POSTGRES_INSTANCE"
$CONTAINER_CMD exec "instance://$POSTGRES_INSTANCE" bash -c "docker-entrypoint.sh postgres -p ${POSTGRES_PORT} &"

for _ in {1..30}; do
    if $CONTAINER_CMD exec "instance://$POSTGRES_INSTANCE" pg_isready -h localhost -p "$POSTGRES_PORT" -U postgres > /dev/null 2>&1; then
        echo -e "${GREEN}PostgreSQL is ready on port ${POSTGRES_PORT}${NC}"
        exit 0
    fi
    sleep 2
done

echo -e "${RED}PostgreSQL failed to start within 60 seconds${NC}"
$CONTAINER_CMD instance stop "$POSTGRES_INSTANCE" 2>/dev/null || true
exit 1
