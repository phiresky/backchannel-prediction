for f in out/*/config.json; do echo $(jq '[.train_output.stats[]|.validation_error]|min' $f) $f; done |sort -r
