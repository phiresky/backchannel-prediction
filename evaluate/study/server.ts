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
let netRatingSegments: string[][];
const preferred = [
    "sw2007B @294.",
    //"sw2476B @13.",
    "sw2396A @381.",
    "sw2491B @387.",
    "sw4774B @130.",
    "sw3038B @250.",
    "sw2325A @61.",
] as string[];
class TwoWayMap<A, B> {
    mapA = new Map<A, B>();
    mapB = new Map<B, A>();
    hasA(x: A) {
        return this.mapA.has(x);
    }
    hasB(x: B) {
        return this.mapB.has(x);
    }
    getA(x: A) {
        return this.mapA.get(x);
    }
    getB(x: B) {
        return this.mapB.get(x);
    }
    set(x: A, y: B) {
        this.mapA.set(x, y);
        this.mapB.set(y, x);
    }
}
const nocaching = { etag: false, maxage: 0 };
const urlMap = new TwoWayMap<number, string>();
const wantedSegCount = 6;
let urlCounter = 0;
function addSecretUrl(realUrl: string) {
    const ext = realUrl.split(".").slice(-1)[0];
    const counter = ++urlCounter;
    urlMap.set(counter, realUrl);
    console.log(counter, "=", realUrl);
    return `data/${counter}.${ext}`;
}
function secretUrlToReal(secretUrl: string) {
    if (secretUrl.startsWith("data/")) secretUrl = secretUrl.substring(4);
    const id = +secretUrl.substr(1).split(".")[0];
    if (urlMap.hasA(id)) {
        return urlMap.getA(id)!;
    } else {
        console.error("not found: " + secretUrl);
        return secretUrl;
    }
}
function dataRewriter(req: express.Request, res: express.Response, next: express.NextFunction) {
    req.url = secretUrlToReal(req.url);
    res.header('Cache-Control', 'no-cache');
    next();
}
function assumeSingleGlob(path: string) {
    const g = glob.sync(path);
    if (g.length !== 1) console.error("cannot find", path);
    return g[0];
}
const meths = ["nn", "truthrandom", "random"];
const r = Random.engines.mt19937().autoSeed(); //.seed(1337);

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

    monosegs = glob.sync(join(__dirname, "data/mono/*.wav"));
    Random.shuffle(r, monosegs);
    monosegs.unshift(...preferred.map(str => assumeSingleGlob(join(__dirname, "data/mono", str + "*.wav"))));
    console.log("we have", preferred.length, "preferred segments, adding", wantedSegCount - preferred.length, "random ones");
    monosegs = monosegs.map(d => join("/mono", basename(d)));
    monosegs = monosegs.slice(0, wantedSegCount);
    netRatingSegments = monosegs
        .map(monoseg => basename(monoseg).split(".")[0])
        .map(monoseg => Random.shuffle(r, meths).map(meth => join(__dirname, "data", meth, monoseg + "*.mp3"))
            .map(g => assumeSingleGlob(g))
            .map(f => f.substr(join(__dirname, "data").length))
        );
    monosegs = monosegs.map(url => addSecretUrl(url));
    console.log(netRatingSegments);
    netRatingSegments = netRatingSegments.map(x => x.map(url => addSecretUrl(url)));
    console.log("loaded", monosegs.length, "monosegs");

    const app = express();
    const server = http.createServer(app);
    server.listen(process.env.PORT || 8000);
    // force compression
    // app.use((req, res, next) => (req.headers['accept-encoding'] = 'gzip', next()))
    // app.use(compression({filter: () => true}));
    app.use("/", express.static(join(__dirname, "build"), nocaching));
    app.use("/data", dataRewriter);
    app.use("/data", express.static(join(__dirname, "data"), nocaching));
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
    @PrimaryGeneratedColumn()
    id: number;
    @ManyToOne(type => Session, session => session.netRatings)
    session: Session;
    @Column()
    segment: string;
    @Column("int", { nullable: true })
    rating: number | null;
    @CreateDateColumn()
    created: Date;
    @Column()
    final: boolean;
}
function initClient(_client: SocketIO.Socket) {
    const client = _client as common.RouletteServerSocket;
    let session = new Session();
    const meta = _client.request;
    session.handshake = JSON.stringify(_client.handshake);
    const sessionPersisted = db.entityManager.persist(session);
    client.on("beginStudy", async (data, callback) => {
        console.log(new Date(), "beginning study");
        session.bcSampleSource = data.bcSampleSource;
        callback({ sessionId: session.id });
    });
    client.on("getData", async (options, callback) => {
        await sessionPersisted;
        const segments = Random.shuffle(r, netRatingSegments.map(choices => Random.sample(r, choices, 1)[0]));
        callback({ bcSamples, monosegs, netRatingSegments: segments, sessionId: session.id});
    });
    client.on("submitBC", async (options, callback) => {
        const pred = new BCPrediction();
        pred.session = session;
        pred.segment = secretUrlToReal(options.segment);
        pred.time = options.time;
        pred.duration = options.duration;
        await db.entityManager.persist(pred);
        console.log("bc", pred.session.id, pred.segment, pred.time);
        callback({});
    });
    client.on("submitNetRatings", async ({ segments, final }, callback) => {
        const entities = segments.map(([segment, rating]) => {
            const x = new NetRating();
            x.rating = rating;
            segment = secretUrlToReal(segment);
            x.segment = segment;
            x.session = session;
            x.final = final;
            if (final) console.log(session.id, "rated", segment, "with", rating);
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
