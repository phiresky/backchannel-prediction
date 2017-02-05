import * as fs from 'fs';

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
    "eval_config": {},
    extract_config,
    train_config
});


const features_std = [
    "get_power",
    "get_pitch"
];
const features_ffv = [...features_std, "get_ffv"];

const rangeround = (x: number) => +x.toFixed(3);

const method_std = ({bcend = 0, span = 1.51, nbcend = Math.min(-2, bcend - span),} = {}) => ({
    "type": "discrete",
    "bc": [bcend - span, bcend].map(rangeround),
    "nbc": [nbcend - span, nbcend].map(rangeround)
});

const method_close = ({span = 1}) => method_std({nbcend: -0.1 - span - 0.1, span});

const method_far = ({span = 1}) => method_std({nbcend: -0.1 - span - 1.3, span});

const make_extract_config = ({input_features = features_ffv, extraction_method = method_std()} = {}) => ({
    input_features, extraction_method,
    "useOriginalDB": true,
    "useWordsTranscript": false,
    "sample_window_ms": 32,
    "outputDirectory": "extract_pfiles_python/out"
});

const make_train_config = ({
    context_ms = 1500, context_stride = 2, layer_sizes = [70, 35] as number[]|[number|null, number][],
    model_function = "lstm_simple",
    epochs = 100
} = {}) => ({
    model_function,
    epochs,
    context_ms,
    context_stride,
    layer_sizes,
    "resume_parameters": null,
    "update_method": "adam",
    "learning_rate": 0.001,
    "l2_regularization": 0.0001,
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
const interesting_layers_dropout: [number|null, number][][] = [
    [[null, 0.2], [50, 0.5], [20, 0.5]],
    [[null, 0.2], [75, 0.5], [40, 0.5]],
    [[null, 0.2], [100, 0.5], [50, 0.5]],
    [[null, 0.2], [125, 0.5], [80, 0.5]],
    [[null, 0.2], [100, 0.5], [50, 0.5], [25, 0.5]],
    [[null, 0.2], [100, 0.5], [70, 0.4], [50, 0.3], [40, 0.2]]
];


const best_layers = model_function => ({
    "feedforward_simple": [
        [75, 40],
        [100, 50, 25],
    ],
    "lstm_simple": [
        [50, 20],
        [100, 20, 100]
    ]
})[model_function];

const mfcc_combos = [
    [...features_std, "get_ffv", "get_mfcc"],
    [...features_std, "get_mfcc"],
    ["get_power", "get_ffv", "get_mfcc"],
    ["get_power", "get_mfcc"]
];
const dropout_name = categoryname => `${categoryname}-${layers.map(([lsize, dropout]) => `${lsize || "inp"}.${dropout.toString().split(".")[1]}`).join("-")}`;


function write_config(category: string, name: string, config: any) {
    const outdir = `configs/finunified/${category}`;
    if (!fs.existsSync(outdir)) fs.mkdirSync(outdir);
    fs.writeFileSync(`configs/finunified/${category}/${name}.json`, JSON.stringify(config, null, '\t'));
}
const interesting_contexts = [0.5, 1.0, 1.5, 2.0];
for (const context of interesting_contexts) {
    const span = context + 0.01;
    const context_ms = Math.round(context * 1000) | 0;
    const extraction_method = method_std({span, bcend: 0, nbcend: Math.min(-2, 0 - span)});
    const extract_config = make_extract_config({extraction_method});
    const train_config = make_train_config({context_ms});
    const name = `lstm-best-context-${context_ms}ms`;
    const config = make_config({
        name, extract_config, train_config
    });
    write_config("vary-context", name, config);
}
const interesting_features = [
    features_std, features_ffv,
    [...features_std, "get_word2vec_v1"],
    [...features_ffv, "get_word2vec_v1"],
    [...features_std, "get_word2vec_dim10"],
    [...features_std, "get_mfcc"],
    ["get_power", "get_ffv", "get_mfcc"],
    ["get_power", "get_ffv"]
];
const interesting_strides = [1, /*default/best = 2, */4];
const interesting_layers_best = [
    [100],
    [50, 20],
    /*default/best = [70, 35]*/
    [100, 50],
    [70, 50, 35]
];
for (const input_features of interesting_features) {
    const extract_config = make_extract_config({input_features});
    const name = `lstm-best-features-${input_features.map(feat => feat.substr(feat.indexOf("_") + 1))}`;
    const config = make_config({name, extract_config});
    write_config("vary-features", name, config);
}
for (const context_stride of interesting_strides) {
    const extract_config = make_extract_config();
    const train_config = make_train_config({context_stride});
    const name = `lstm-best-stride-${context_stride}`;
    const config = make_config({
        name, extract_config, train_config
    });
    write_config("vary-stride", name, config);
}

for (const layer_sizes of interesting_layers_best) {
    const extract_config = make_extract_config();
    const train_config = make_train_config({layer_sizes});
    const name = `lstm-best-layers-${layer_sizes.join("-")}`;
    const config = make_config({name, extract_config, train_config});
    write_config("vary-layers", name, config);
}

/*for (const model_function of ["lstm_simple", "feedforward_simple"]) {
 const categoryname0 = {feedforward: "ff", lstm: "lstm"}[model_function.split("_")[0]];
 for (const input_features of mfcc_combos) {
 const categoryname = categoryname0 + "-" + input_features.map(feat => feat.split("_")[1]);
 for (const layer_sizes of best_layers(model_function)) {
 const name = `${categoryname}-${layer_sizes.join("-")}`;
 const config = make_config({
 name: name,// + '-l2reg',
 extract_config: make_extract_config({input_features, extraction_method: method_std()}),
 train_config: {
 ...make_train_config({layer_sizes, model_function, epochs: 100}),
 update_method: "adam",
 learning_rate: 0.001,
 //l2_regularization: 0.0001
 },
 });
 const outdir = `configs/${categoryname}`;
 if (!fs.existsSync(outdir)) fs.mkdirSync(outdir);
 fs.writeFileSync(`configs/${categoryname}/${name}.json`, JSON.stringify(config, null, '\t'));
 }
 }
 }*/
