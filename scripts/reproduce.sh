#!/usr/bin/bash

set -e

if [[ ! -d configs/finunified ]]; then
    echo "you must run this script from the repository root"
    exit 1
fi
echo "regenerating configs"
rm -r configs/finunified/vary-*
ts-node --fast configs/meta_config_generator.ts

for config in configs/finunified/vary-*/*.json; do
    echo "reproducing results for config $config"
    python -m trainNN.train "$config"
    gitversion="$(git describe --dirty)"
    configname="$(jq -r .name "$config")"
    version="$gitversion:$configname"
    echo "evaluation $version"
    python -m evaluate.evaluate "trainNN/out/$version/config.json"
done

echo "See the Section 'Evaluation Visualizer' in the readme to see the results"
