Code for my bachelor thesis and the corresponding paper

More documentation to follow :)


The final configurations are in configs/finunified. All of these are generated with configs/meta_config_generator.ts


## Meta config generator

Generates configurations from a set of combinations 

Run this from the project root:

    ts-node --fast configs/meta_config_generator.ts
   

## Web Visualizer

Server is in /web_vis/py/

Run this from the project root:

    python -m web_vis.py.server extract/config.json

Client is in /web_vis/ts/

Run this from the folder /web_vis/ts/

    yarn run dev

This will start a webserver serving the client at <http://localhost:3000>, which will connect to the server via websockets at localhost:8765.

## Extraction

## Training

