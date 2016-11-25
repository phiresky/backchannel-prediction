import {action, observable, reaction, asReference} from 'mobx';
import * as c from './client';
import * as util from './util';

export type ClientMessage = {
    type: "getConversations"
} | {
    type: "getFeatures",
    conversation: ConversationID
} | {
    type: "getFeature",
    conversation: ConversationID,
    feature: FeatureID
}
type GetConversationsResponse = ConversationID[];
type GetFeatureResponse = c.Feature;
type GetFeaturesResponse = {
    defaults: FeatureID[],
    optional: FeatureID[]
}
type TypedArrayTypes = 'float32' | 'int16';
function TypedArrayOf(type: TypedArrayTypes) {
    switch(type) {
        case 'float32': return Float32Array;
        case 'int16': return Int16Array;
        default: throw Error("unknown type");
    }
}

class LulPromise<T> {
    @observable data: T|null;
    constructor(data: T|null) {
        this.data = asReference(data);
    }
    toPromise(): Promise<T> {
        if(this.data !== null) return Promise.resolve(this.data);
        else return new Promise(resolve => reaction<T>("toPromise", () => this.data!, resolve));
    }
}

export interface ConversationID extends String {
    __typeBrand: "ConversationID"
}
export interface FeatureID extends String {
    __typeBrand: "FeatureID"
}
export interface ServerMessage {
    id: number;
    data: any;
}
export class SocketManager {
    socket: WebSocket;
    nextMessageID = 1;
    listeners = new Map<number, (msg: ServerMessage) => void>();
    nextBinaryFrameListener: null | ((frame: ArrayBuffer) => void);
    constructor(private server: string) {
        
    }
    async socketOpen() {
        if(!this.socket || this.socket.readyState === this.socket.CLOSED) {
            this.socket = new WebSocket(this.server);
            this.socket.binaryType = "arraybuffer";
            this.socket.onmessage = this.onSocketMessage.bind(this);
        }
        if(this.socket.readyState === this.socket.OPEN) return;
        await new Promise(resolve => this.socket.addEventListener("open", e => resolve()));
    }
    onSocketMessage(event: MessageEvent) {
        if(event.data instanceof ArrayBuffer) {
            console.log("RECEIVING", event.data);
            if(!this.nextBinaryFrameListener) throw Error("got unexpected binary frame");
            this.nextBinaryFrameListener(event.data);
            this.nextBinaryFrameListener = null;
        } else {
            const msg: ServerMessage = JSON.parse(event.data);
            console.log("RECEIVING", msg);
            const listener = this.listeners.get(msg.id);
            if(!listener) throw Error("unexpected message: " + msg.id);
            listener(msg);
        }
    }
    waitForBinaryFrame() {
        if(this.nextBinaryFrameListener) throw Error("tried to receive two binary frames at the same time");
        return new Promise<ArrayBuffer>(resolve => this.nextBinaryFrameListener = resolve);
    }
    async sendMessage(message: ClientMessage): Promise<ServerMessage> {
        await this.socketOpen();
        const id = this.nextMessageID++;
        (message as any).id = id;
        console.log(`SENDING [${id}]: `, message);
        this.socket.send(JSON.stringify(message));
        return new Promise<ServerMessage>((resolve, reject) => {
            this.listeners.set(id, resolve);
        });
    }
    
    async getConversationsRemote(): Promise<GetConversationsResponse> {
        const response = await this.sendMessage({type: "getConversations"});
        return response.data as GetConversationsResponse;
    }
    getConversations(): LulPromise<GetConversationsResponse> {
        return lulCache("getConversations", this, this.getConversationsRemote, Array.from(arguments));
    }

    async getFeaturesRemote(conversation: ConversationID): Promise<GetFeaturesResponse> {
        const response = await this.sendMessage({type: "getFeatures", conversation});
        return response.data as GetFeaturesResponse;
    }
    getFeatures(conversation: ConversationID): LulPromise<GetFeaturesResponse> {
        return lulCache("getFeatures", this, this.getFeaturesRemote, Array.from(arguments));
    }
    async getFeatureRemote(conversation: ConversationID, featureID: FeatureID): Promise<GetFeatureResponse> {
        const response = await this.sendMessage({type: "getFeature", conversation, feature: featureID});
        const feature = response.data as GetFeatureResponse;
        if(feature.data === null) {
            if(!c.isNumFeature(feature)) throw Error("wat");
            const buffer = await this.waitForBinaryFrame();
            feature.data = new (TypedArrayOf(feature.dtype))(buffer);
        }
        return feature;
    }
    getFeature(conversation: ConversationID, featureID: FeatureID): LulPromise<GetFeatureResponse> {
        return lulCache("getFeature", this, this.getFeatureRemote, Array.from(arguments));
    }
}

const cache = new Map<string, LulPromise<any>>();
function lulCache<T>(name: string, _this: any, fn: (...args: any[]) => Promise<T>, args: any[]) {
    const key = JSON.stringify(util.toDeterministic([...args, name]));
    if(!cache.has(key)) {
        const lul = new LulPromise<T>(null);
        cache.set(key, lul);
        (fn.apply(_this, args) as Promise<T>).then(action("received"+name, (result:T) => lul.data = result));
    }
    return cache.get(key)!;
}