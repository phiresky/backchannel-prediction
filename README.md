Code for my bachelor thesis and the corresponding paper

More documentation to follow :)


The final configurations are in configs/finunified. All of the ones in `vary-*` are generated with configs/meta_config_generator.ts

## Setup

### Get the data

See [data/README.md](data/README.md) for more details.

### Build Janus

```bash
cd janus
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo python setup.py develop
```

## Meta config generator

Generates configurations from a set of combinations 

Run this from the project root:

    ts-node --fast configs/meta_config_generator.ts
   
The best network configuration according to the objective evaluation is

    configs/finunified/vary-features/lstm-best-features-power,pitch,ffv,word2vec_dim30.json


## Data Visualizer

Server is in /web_vis/py/

Run this from the project root:

    python -m web_vis.py.server extract/config.json

Client is in /web_vis/ts/

Run this from the folder /web_vis/ts/

    yarn run dev

This will start a webserver serving the client at <http://localhost:3000>, which will connect to the server via websockets at localhost:8765.

## Extraction

The main script for extraction is extract/readDB.py. Run it via

    export JOBS=4 # run in parallel
    python -m extract.readDB configs/...
   
Note that the extraction will also be run automatically when before training when necessary

## Training

## Evaluation Visualizer

Build it and run the server

    cd evaluate/plot
    yarn
    yarn run dev

Then go to <http://localhost:8080/evaluate/plot/dist/>