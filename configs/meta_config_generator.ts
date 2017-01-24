const make_config = ({extract_config = make_extract_config(), train_config = make_train_config()} = {}) => ({
    "name": "lstm-continued-sgd",
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


const method_std = ({bcend = -0.1, nbcend = -1.9, span = 1}) => ({
    "type": "discrete",
    "bc": [bcend - span, bcend],
    "nbc": [nbcend - span, nbcend]
});

const method_close = ({span = 1}) => method_std({nbcend: -0.1 - span - 0.1, span});

const method_far = ({span = 1}) => method_std({nbcend: -0.1 - span - 1.3, span});

const make_extract_config = ({input_features = features_std, extraction_method = method_std} = {}) => ({
    input_features, extraction_method,
    "useOriginalDB": true,
    "useWordsTranscript": false,
    "sample_window_ms": 32,
    "outputDirectory": "extract_pfiles_python/out"
});

const make_train_config = ({} = {}) => ({
    "model_function": "lstm_simple",
    "resume_parameters": "trainNN/out/v042-unified-dirty:lstm-single-pre-test/epoch-002.pkl",
    "context_ms": 800,
    "context_stride": 2,
    "update_method": "sgd",
    "learning_rate": 0.5,
    "num_labels": 2,
    "batch_size": 250,
    "epochs": 100,
    "gaussian": false,
    "layer_sizes": [100, 50],
    "output_type": "single"
});

console.log(make_config());