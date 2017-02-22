import * as http from 'http';
import * as express from 'express';
import * as webpackMiddleware from 'webpack-dev-middleware';
import * as webpackConfig from './webpack.config';
import * as io from 'socket.io';
import * as common from './common';
import { join, basename } from 'path';
import * as glob from 'glob';
import * as Random from 'random-js';
import * as compression from 'compression';
import {
    createConnection, Connection,
} from 'typeorm';
import { openDatabase, Session, BCPrediction, NetRating } from './db';

import "reflect-metadata";
let db: Connection;
const nocaching = { etag: false, maxage: 0 };

async function listen() {
    db = await createConnection({
        driver: {
            type: "sqlite",
            storage: join(__dirname, "db.sqlite")
        },
        entities: [
            Session, BCPrediction, NetRating
        ],
        autoSchemaSync: true
    })

    const app = express();
    const server = http.createServer(app);
    server.listen(process.env.PORT || 8000);
    app.use(compression());
    app.get("/pcqxnugylresibwhwmzv/ratings.json", async (req, res) => {
        res.setHeader('Access-Control-Allow-Origin', '*');
        res.setHeader('Content-Type', 'application/json');
        const resp = await db.entityManager.createQueryBuilder(NetRating, "rating")
            .innerJoinAndSelect("rating.session", "session")
            .getMany();
        res.send(JSON.stringify(resp, (k, v) => k === 'handshake' ? JSON.parse(v) : v));
    });
    app.get("/pcqxnugylresibwhwmzv/sessions.json", async (req, res) => {
        res.setHeader('Access-Control-Allow-Origin', '*');
        res.setHeader('Content-Type', 'application/json');
        const resp = await db.entityManager.find(Session);
        res.send(JSON.stringify(resp, (k, v) => k === 'handshake' ? JSON.parse(v) : v));
    });

}
listen();
