#!/usr/bin/env bash
MASTER=$( hostname )
PORT=6379
PYTHONPATH=/home/pfahler/deep-learning-on-fact/code/deep-ensembles-v2 ray start --redis-port=${PORT} --memory 45000000000 --object-store-memory 10000000000 --num-cpus 8 --head
SLAVES="ls8gpu04.local ls8gpu05.local ls8gpu06.local ls8gpu08.local ls8gpu09.local"
for DEVICE in $SLAVES; do
    ssh ${DEVICE} "source /opt/anaconda3/bin/activate && conda activate arxiv && ray stop && PYTHONPATH=/home/pfahler/deep-learning-on-fact/code/deep-ensembles-v2 ray start --address=${MASTER}:${PORT} --object-store-memory 10000000000 "
done
