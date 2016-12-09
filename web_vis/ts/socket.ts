import * as mobx from "mobx";
import * as c from "./client";
import * as util from "./util";
import * as Data from "./Data";
import { autobind } from "core-decorators";
import { NumFeatureCommon, Utterances, Highlights, NumFeature, Feature, FeatureID, ConversationID } from "./features";

export type ClientMessage = {
    type: "getConversations"
} | {
        type: "getFeatures",
        conversation: ConversationID
    } | {
        type: "getFeature",
        conversation: ConversationID,
        feature: FeatureID
    } | {
        type: "echo"
    };
type GetConversationsResponse = { [name: string]: ConversationID[] };
type GetFeatureResponse = FeatureReceive;
export type CategoryTreeElement = { name: string, children: CategoryTreeElement[] } | string;
export type GetFeaturesResponse = {
    categories: CategoryTreeElement[];
    defaults: FeatureID[][];
}

export class LulPromise<T> implements PromiseLike<T> {
    @mobx.observable data: T | null = mobx.asReference(null);
    promise: Promise<T>;
    constructor(arg: T | Promise<T>) {
        if (typeof arg === "object" && typeof (arg as any)["then"] !== "function") {
            mobx.runInAction("instaresolve", () => this.data = arg as T);
            this.promise = Promise.resolve(this.data);
        } else {
            this.promise = arg as Promise<T>;
            this.promise.then(mobx.action("fromPromise", (result: T) => this.data = result));
        }
    }
    then<R>(resolve: (t: T) => R, reject?: (error: any) => any): LulPromise<R> {
        if (this.data) {
            // resolve instantly without waiting for call stack to empty
            const nextVal = resolve(this.data);
            const nextLul = new LulPromise(Promise.resolve(nextVal));
            if (typeof nextVal === "object" && typeof (nextVal as any)["then"] !== "function") {
                mobx.runInAction("then()", () => nextLul.data = nextVal);
            }
            return nextLul;
        } else {
            return new LulPromise(this.promise.then(resolve, reject));
        }
    }
}

export type ServerMessage = { id: number, error: undefined, data: any };
export type ServerError = { id: number, error: string };


export type NumFeatureReceive = NumFeatureCommon & {
    typ: "FeatureType.SVector" | "FeatureType.FMatrix",
    shape: number[],
    dtype: Data.TypedArrayType
};
export type FeatureReceive = NumFeatureReceive | Utterances | Highlights;
declare var TextDecoder: any;
export type BinaryFrameMeta = {
    conversation: ConversationID,
    feature: FeatureID,
    byteOffset: number
}
let frameStart = 0;
export class SocketManager {
    socket: WebSocket;
    nextMessageID = 1;
    listeners = new Map<number, { resolve: (msg: ServerMessage) => void, reject: (reason: any) => void }>();
    queue = [] as (() => void)[];
    constructor(private server: string) {

    }
    async socketOpen() {
        if (!this.socket || this.socket.readyState === this.socket.CLOSED) {
            this.socket = new WebSocket(this.server);
            this.socket.binaryType = "arraybuffer";
            this.socket.onmessage = this.onSocketMessage.bind(this);
        }
        if (this.socket.readyState === this.socket.OPEN) return;
        await new Promise(resolve => this.socket.addEventListener("open", e => resolve()));
    }
    @autobind @mobx.action
    squashQueue() {
        frameStart = performance.now();
        while (this.queue.length > 0) {
            if (performance.now() - frameStart > c.globalConfig.minRenderDelayMS) {
                console.log(`time is up; waiting until next frame for remaining ${this.queue.length} elements`);
                return requestAnimationFrame(this.squashQueue);
            }
            this.queue.shift() !();
        }
        return 0;
    }
    async onSocketMessage(event: MessageEvent) {
        if (event.data instanceof ArrayBuffer) {
            if (this.queue.length === 0) {
                requestAnimationFrame(this.squashQueue);
            }
            this.queue.push(() => {
                const buffer = event.data;
                const metaLength = new Int32Array(buffer, 0, 1)[0];
                const metaBuffer = new Uint8Array(buffer, 4, metaLength);
                const metaText = new TextDecoder("utf-8").decode(metaBuffer);
                const meta = JSON.parse(metaText) as BinaryFrameMeta;
                console.log("RECEIVING BUF", meta);
                const feature = this.getFeature(meta.conversation, meta.feature).data;
                if (!feature) throw Error("received feature data before feature: " + meta.feature);
                if (!c.isNumFeature(feature)) throw Error("wat2");
                feature.data.setData(meta.byteOffset, buffer, metaLength + 4, buffer.byteLength - metaLength - 4);
            });
        } else {
            const msg: ServerMessage | ServerError = JSON.parse(event.data);
            console.log("RECEIVING", msg);
            const listener = this.listeners.get(msg.id);
            if (!listener) throw Error("unexpected message: " + msg.id);
            if (msg.error !== undefined) listener.reject(msg.error);
            else listener.resolve(msg as ServerMessage);
        }
    }
    async sendMessage(message: ClientMessage): Promise<ServerMessage> {
        await this.socketOpen();
        const id = this.nextMessageID++;
        (message as any).id = id;
        console.log(`SENDING [${id}]: `, message);
        this.socket.send(JSON.stringify(message));
        return new Promise<ServerMessage>((resolve, reject) => {
            this.listeners.set(id, { resolve, reject });
        });
    }

    async getConversationsRemote(): Promise<GetConversationsResponse> {
        const response = await this.sendMessage({ type: "getConversations" });
        return response.data as GetConversationsResponse;
    }
    getConversations(): LulPromise<GetConversationsResponse> {
        return lulCache("getConversations", this, this.getConversationsRemote, Array.from(arguments));
    }

    async getFeaturesRemote(conversation: ConversationID): Promise<GetFeaturesResponse> {
        const response = await this.sendMessage({ type: "getFeatures", conversation });
        return response.data as GetFeaturesResponse;
    }
    getFeatures(conversation: ConversationID): LulPromise<GetFeaturesResponse> {
        return lulCache("getFeatures", this, this.getFeaturesRemote, Array.from(arguments));
    }
    async echo() {
        var before = performance.now();
        await this.sendMessage({ type: "echo" })
        console.log("roundtrip took", performance.now() - before, "ms");
    }
    async getFeatureRemote(conversation: ConversationID, featureID: FeatureID): Promise<Feature> {
        const response = await this.sendMessage({ type: "getFeature", conversation, feature: featureID });
        const feature = response.data as GetFeatureResponse;
        feature.name = featureID;
        if (feature.typ === "FeatureType.FMatrix" || feature.typ === "FeatureType.SVector") {
            return mobx.extendObservable(feature, {
                data: new Data.TwoDimensionalArray(feature.dtype, feature.typ === "FeatureType.FMatrix" ? feature.shape as any : [feature.shape[0], 1])
            }) as NumFeature;
        }
        return feature as Highlights;
    }
    getFeature(conversation: ConversationID, featureID: FeatureID): LulPromise<Feature> {
        return lulCache("getFeature", this, this.getFeatureRemote, Array.from(arguments));
    }
}

const cache = new util.LazyHashMap<any, LulPromise<any>>();
function lulCache<T>(name: string, _this: any, fn: (...args: any[]) => Promise<T>, args: any[]) {
    const key = [...args, name];

    if (!cache.has(key)) {
        const lul = new LulPromise(fn.apply(_this, args) as Promise<T>);
        cache.set(key, lul);
    }
    return cache.get(key) !;
}