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
    createConnection, Entity, Column,
    PrimaryColumn, PrimaryGeneratedColumn, Connection, OneToOne, OneToMany, ManyToOne,
    CreateDateColumn
} from 'typeorm';
import "reflect-metadata";
let db: Connection;
let bcSamples: common.BCSamples[];
let monosegs: string[];
let netRatingSegments: string[];
const preferred = [
    "sw2007B @294.",
    "sw2476B @13."
];
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
    bcSamples = glob.sync(join(__dirname, "data/BC", "*/"))
        .map(dir => ({ name: basename(dir), samples: glob.sync(join(dir, "*.wav")).map(x => join("data/BC", basename(dir), basename(x))) }));
    console.log("loaded", bcSamples.length, "bc sample files");

    const r = Random.engines.mt19937().seed(1337);
    monosegs = glob.sync(join(__dirname, "data/mono/*.wav")).map(d => join("data/mono", basename(d)));
    Random.shuffle(r, monosegs);
    monosegs = monosegs.slice(0, 10);
    netRatingSegments = monosegs
        .map(monoseg => basename(monoseg).split(".")[0])
        .map(monoseg => join(__dirname, "data/nn", monoseg + "*.wav"))
        .map(g => glob.sync(g)[0])
        .map(d => join("data/nn", basename(d)));
    console.log("loaded", monosegs.length, "monosegs");

    const app = express();
    const server = http.createServer(app);
    server.listen(process.env.PORT || 8000);
    // force compression
    // app.use((req, res, next) => (req.headers['accept-encoding'] = 'gzip', next()))
    // app.use(compression({filter: () => true}));
    app.use("/", express.static(join(__dirname, "build")));
    app.use("/data", express.static(join(__dirname, "data")));
    const socket = io(server);

    socket.on('connection', initClient);
}

@Entity()
class Session {
    @PrimaryGeneratedColumn()
    id: number;
    @Column({ nullable: true })
    bcSampleSource: string | null = null;
    @OneToMany(type => BCPrediction, bc => bc.session)
    bcs: BCPrediction[];
    @CreateDateColumn()
    created: Date;
    @Column()
    handshake: string;
    @OneToMany(type => NetRating, rating => rating.session)
    netRatings: NetRating[];
    @Column({ nullable: true })
    comment: string;
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
    @Column()
    duration: number;
    @CreateDateColumn()
    created: Date;
}
@Entity()
export class NetRating {
    @ManyToOne(type => Session, session => session.netRatings)
    session: Session;
    @PrimaryColumn()
    segment: string;
    @Column({ nullable: true })
    rating: number | null;
    @CreateDateColumn()
    created: Date;
}
function initClient(_client: SocketIO.Socket) {
    const client = _client as common.RouletteServerSocket;
    let session = new Session();
    const meta = _client.request;
    session.handshake = JSON.stringify(_client.handshake);
    db.entityManager.persist(session);
    client.on("beginStudy", async (data, callback) => {
        console.log(new Date(), "beginning study");
        session.bcSampleSource = data.bcSampleSource;
        await db.entityManager.persist(session);
        callback({ sessionId: session.id });
    });
    client.on("getData", (options, callback) => {
        callback({ bcSamples, monosegs, netRatingSegments });
    });
    client.on("submitBC", async (options, callback) => {
        const pred = new BCPrediction();
        pred.session = session;
        pred.segment = options.segment;
        pred.time = options.time;
        pred.duration = options.duration;
        await db.entityManager.persist(pred);
        console.log("bc", pred.session.id, pred.segment, pred.time);
        callback({});
    });
    client.on("submitNetRatings", async (options, callback) => {
        const entities = options.map(([segment, rating]) => {
            const x = new NetRating();
            x.rating = rating;
            x.segment = segment;
            x.session = session;
            console.log(session.id, "rated", segment, "with", rating);
            return x;
        });
        await db.entityManager.persist(entities);
        callback({});
    });
    client.on("comment", async (options, callback) => {
        session.comment = options;
        console.log(session.id, "comment", options);
        await db.entityManager.persist(session);
        callback({});
    })
}

listen();
