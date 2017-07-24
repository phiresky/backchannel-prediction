Code for my bachelor thesis and the corresponding paper

The final configurations are in [configs/finunified](configs/finunified). All of the ones in `vary-*` are generated with configs/meta_config_generator.ts

## Demo

1. Sample on the evaluation data set: https://streamable.com/0woc
2. Another sample on the evaluation data set: https://streamable.com/eubh
3. Live system demo (from microphone input): https://streamable.com/dycu1

Demo docker container:

    sudo docker run -p 3000:3000 -p 8765:8765 -ti phiresky/backchanneler-live-demo

Then open <http://localhost:3000> in the browser.

Includes the live demo (microphone input) and some sample tracks from the Switchboard data set: sw2249, sw2258, sw2411, sw2485, sw273,  sw2807, w2254, sw2297, sw2432, sw2606, sw2762, sw4193.

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

## Reproducing the results of the paper

You can reproduce the results of the paper using the script [scripts/reproduce.sh](scripts/reproduce.sh).

Note that this may take a long time (~3h to extract the data (only once), ~2h to train one LSTM, 1h to evaluate it).

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
   
Note that the extraction will also be run automatically when before training when necessary, with all the results being cached. The `data/cache` folder will grow to about 10-20 GByte.

## Training

## Evaluation Visualizer

Build it and run the server

    cd evaluate/plot
    yarn
    yarn run dev

Then go to <http://localhost:8080/evaluate/plot/dist/>

## Technical details

You can see more information in `Section 6: Implementation` of my bachelor's thesis, see here: https://github.com/phiresky/bachelor-thesis/blob/master/build/thesis.pdf
