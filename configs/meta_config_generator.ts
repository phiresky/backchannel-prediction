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

const rangeround = (x: number) => +x.toFixed(3);

const method_std = ({bcend = -0.1, nbcend = bcend - 1.8, span = 1} = {}) => ({
    "type": "discrete",
    "bc": [bcend - span, bcend].map(rangeround),
    "nbc": [nbcend - span, nbcend].map(rangeround)
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

const make_train_config = ({context_ms = 800, context_stride = 2, layer_sizes = [100, 50],
    model_function = "feedforward_simple",
    epochs = 200
} = {}) => ({
    model_function,
    epochs,
    context_ms,
    context_stride,
    layer_sizes,
    "resume_parameters": null,
    "update_method": "sgd",
    "learning_rate": 0.7,
    "num_labels": 2,
    "batch_size": 250,
    "gaussian": false,
    "output_type": "single"
});

const interesting_layers_normal = [
    [200], [50],
    [100, 50], [75, 75], [50, 100],
    [75, 40], [60, 60],
    [50, 20], [35, 35],
    [100, 50, 25], [100, 50, 50], [100, 20, 100], [70, 50, 40, 30]
];
const interesting_layers_dropout = [
    [[null, 0.2], [75, 0.5], [40, 0.5]],
    [[null, 0.2], [100, 0.5], [50, 0.5]],
    [[null, 0.2], [125, 0.5], [80, 0.5]],
    [[null, 0.2], [100, 0.5], [50, 0.5], [25, 0.5]],
    [[null, 0.2], [100, 0.5], [70, 0.4], [50, 0.3], [40, 0.2]]
]
for (const layers of interesting_layers_dropout) {
    const categoryname = "lstm-ffv-dropout";
    const name = `${categoryname}-${layers.map(([lsize, dropout]) => `${lsize||"inp"}.${dropout.toString().split(".")[1]}`).join("-")}`;
    const config = make_config({
        name,
        extract_config: make_extract_config({input_features: features_ffv, extraction_method: method_std()}),
        train_config: {
            ...make_train_config({layer_sizes: layers, model_function: "lstm_dropout", epochs: 100}),
            update_method: "adam",
            learning_rate: 0.001
        },
    });
    const outdir = `configs/${categoryname}`;
    if (!fs.existsSync(outdir)) fs.mkdirSync(outdir);
    fs.writeFileSync(`configs/${categoryname}/${name}.json`, JSON.stringify(config, null, '\t'));
}