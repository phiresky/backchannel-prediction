#!/usr/bin/bash

set -e

if [[ ! -d configs/finunified ]]; then
    echo "you must run this script from the repository root"
    exit 1
fi
echo "regenerating configs"
rm -r configs/finunified/*
ts-node --fast configs/meta_config_generator.ts

for config in configs/finunified/vary-*/*.json; do
    echo "reproducing results for config $config"
    python -m trainNN.train "$config"
done