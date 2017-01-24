const fs = require('fs');

const make_config = ({name, extract_config = make_extract_config(), train_config = make_train_config()}) => ({
    name,
    "paths": {
        "databasePrefix": "data/db/all240302",
        "adc": "data/adc",
        "conversations": {
            "validate": "data/conversations.valid",
            "train": "data/conversations.train",
            "eval": "data/conversations.eval"
        },
        "backchannels": "data/backchannels-top200.txt",
        "originalSwbTranscriptions": "data/swb_ms98_transcriptions"
    },
    "eval_config": {
        "prediction_offset": 0.1
    },
    extract_config,
    train_config
});


const features_std = [
    "get_power",
    "get_pitch"
];
const features_ffv = [...features_std, "get_ffv"];


const method_std = ({bcend = -0.1, nbcend = -1.9, span = 1} = {}) => ({
    "type": "discrete",
    "bc": [bcend - span, bcend],
    "nbc": [nbcend - span, nbcend]
});

const method_close = ({span = 1}) => method_std({nbcend: -0.1 - span - 0.1, span});

const method_far = ({span = 1}) => method_std({nbcend: -0.1 - span - 1.3, span});

const make_extract_config = ({input_features = features_std, extraction_method = method_std()} = {}) => ({
    input_features, extraction_method,
    "useOriginalDB": true,
    "useWordsTranscript": false,
    "sample_window_ms": 32,
    "outputDirectory": "extract_pfiles_python/out"
});

const make_train_config = ({context_ms = 800, context_stride = 2, layer_sizes = [100, 50]} = {}) => ({
    "model_function": "feedforward_simple",
    "resume_parameters": null,
    context_ms,
    context_stride,
    "update_method": "sgd",
    "learning_rate": 0.7,
    "num_labels": 2,
    "batch_size": 250,
    "epochs": 200,
    "gaussian": false,
    layer_sizes,
    "output_type": "single"
});

const interesting_layers = [
    [200], [50],
    [100, 50], [75, 75], [50, 100],
    [75, 40], [60, 60],
    [50, 20], [35, 35],
    [100, 50, 25], [100, 50, 50], [100, 20, 100], [70, 50, 40, 30]
];
for(const layers of interesting_layers) {
    const name = `simple-ff-${layers.join("-")}`;
    const config = make_config({name, train_config: make_train_config({layer_sizes: layers})});
    fs.writeFileSync(`configs/simple-ff/${name}.json`, JSON.stringify(config, null, '\t'));
}