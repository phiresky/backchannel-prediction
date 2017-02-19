import * as http from 'http';
import * as express from 'express';
import * as webpackMiddleware from 'webpack-dev-middleware';
import * as webpackConfig from './webpack.config';
import * as io from 'socket.io';
import * as common from './common';
import { join, basename } from 'path';
import * as glob from 'glob';
import {
    createConnection, Entity, Column,
    PrimaryColumn, PrimaryGeneratedColumn, Connection, OneToOne, OneToMany, ManyToOne,
    CreateDateColumn
} from 'typeorm';
import "reflect-metadata";
let db: Connection;
let bcSamples: common.BCSamples[];
let monosegs: string[];

async function listen() {
    db = await createConnection({
        driver: {
            type: "sqlite",
            storage: join(__dirname, "db.sqlite")
        },
        entities: [
            Session, BCPrediction
        ],
        autoSchemaSync: true
    })
    bcSamples = glob.sync(join(__dirname, "data/BC", "*/"))
        .map(dir => ({ name: basename(dir), samples: glob.sync(join(dir, "*.wav")).map(x => join("data/BC", basename(dir), basename(x))) }));
    console.log("loaded", bcSamples.length, "bc sample files");

    monosegs = glob.sync(join(__dirname, "data/monosegs/*.wav")).map(d => join("data/monosegs", basename(d)));
    console.log("loaded", monosegs.length, "monosegs");

    const app = express();
    const server = http.createServer(app);
    server.listen(process.env.PORT || 8000);
    app.use("/", express.static(join(__dirname, "build")));
    app.use("/data", express.static(join(__dirname, "data")));
    const socket = io(server);

    socket.on('connection', initClient);
}

@Entity()
class Session {
    @PrimaryGeneratedColumn()
    id: number;
    @Column()
    bcSampleSource: string;
    @OneToMany(type => BCPrediction, bc => bc.session)
    bcs: BCPrediction[];
    @CreateDateColumn()
    created: Date;
}
@Entity()
export class BCPrediction {
    @PrimaryGeneratedColumn()
    id: number;
    @ManyToOne(type => Session, session => session.bcs)
    session: Session;
    @Column()
    segment: string;
    @Column()
    time: number;
    @CreateDateColumn()
    created: Date;
}
function initClient(_client: SocketIO.Socket) {
    const client = _client as common.RouletteServerSocket;
    let session: Session;
    client.on("beginStudy", async (data, callback) => {
        console.log(new Date(), "beginning study");
        session = new Session();
        session.bcSampleSource = data.bcSampleSource;
        await db.entityManager.persist(session);
        callback({sessionId: session.id});
    });
    client.on("getBCSamples", (options, callback) => {
        callback(bcSamples);
    });
    client.on("getMonosegs", (options, callback) => callback(monosegs));
    client.on("submitBC", async (options, callback) => {
        const pred = new BCPrediction();
        pred.session = session;
        pred.segment = options.segment;
        pred.time = options.time;
        await db.entityManager.persist(pred);
        console.log("bc", pred);
        callback({});
    });
}

listen();
