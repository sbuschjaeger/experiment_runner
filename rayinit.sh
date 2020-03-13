#!/usr/bin/env bash
MASTER=$( hostname )
PORT=6379
ray start --head --redis-port=${PORT}
SLAVES = "ls8gpu04 ls8gpu05 ls8gpu06 ls8gpu08 ls8gpu09"
for DEVICE in $DEVICES; do
    ssh ${DEVICE} "source /opt/anaconda3/bin/activate && conda activate ray && ray start --address=${MASTER}:${PORT}"
done