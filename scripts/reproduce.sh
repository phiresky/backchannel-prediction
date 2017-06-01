#!/usr/bin/bash

set -e

if [[ ! -d configs/finunified ]]; then
    echo "you must run this script from the repository root"
    exit 1
fi

echo "See the Section 'Evaluation Visualizer' in the readme to see the results"


echo "regenerating configs"
rm -r configs/finunified/vary-*
ts-node --fast configs/meta_config_generator.ts

getOutVersion() {
    config=$1
    gitversion="$(git describe --dirty)"
    configname="$(jq -r .name "$config")"
    echo "$gitversion:$configname"
}

bestconfig="configs/finunified/vary-features/lstm-best-features-power,pitch,ffv,word2vec_dim30.json"

echo "reproducing best config $bestconfig"
python -m trainNN.train "$bestconfig"
bestversion="$(getOutVersion "$bestconfig")"
echo "$bestversion"
python -m evaluate.evaluate "trainNN/out/$bestversion/config.json" allmargins

for config in configs/finunified/vary-*/*.json; do
    if [[ $config == $bestconfig ]]; then continue; fi
    echo "reproducing results for config $config"
    python -m trainNN.train "$config"
    version="$(getOutVersion $config)"
    echo "evaluation $version"
    python -m evaluate.evaluate "trainNN/out/$version/config.json" bestmargin
done

