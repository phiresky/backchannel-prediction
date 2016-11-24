import {action, observable} from 'mobx';
import * as c from './client';

export type ClientMessage = {
    type: "loadConversation",
    name: string
} | {
    type: "getConversations"
}
export type ServerMessage = {
    type: "getConversations",
    data: string[]
} | {
    type: "getFeature",
    data: c.Feature
} | {
    type: "done"
}
type TypedArrayTypes = 'float32' | 'int16';
function TypedArrayOf(type: TypedArrayTypes) {
    switch(type) {
        case 'float32': return Float32Array;
        case 'int16': return Int16Array;
        default: throw Error("unknown type");
    }
}

export class SocketManager {
    socket: WebSocket;
    features = new Map<string, c.Feature>();
    @observable
    conversations = [] as string[];
    nextBinaryFrameBelongsTo: any;

    constructor(private gui: c.GUI, server: string) {
        this.socket = new WebSocket(server);
        this.socket.binaryType = "arraybuffer";
        this.socket.onmessage = this.onSocketMessage.bind(this);
        this.socket.onopen = this.onSocketOpen.bind(this);
    }
    loadConversation(conversation: string) {
        this.sendMessage({type: "loadConversation", name: conversation});
    }
    sendMessage(message: ClientMessage) {
        console.log("SENDING: ", message);
        this.socket.send(JSON.stringify(message));
    }

    onSocketOpen(event: Event) {
        this.sendMessage({type: "getConversations"});
        this.gui.onSocketOpen();
    }

    @action onSocketMessage(event: MessageEvent) {
        if(event.data instanceof ArrayBuffer) {
            console.log("RECEIVING", event.data);
            const feature = this.nextBinaryFrameBelongsTo;
            if(!feature) throw Error("received unanticipated binary frame");
            feature.data = new (TypedArrayOf(feature.dtype))(event.data);
            this.features.set(feature.name, feature);
            this.gui.onFeatureReceived(feature);
            return;
        }
        const data: ServerMessage = JSON.parse(event.data);
        console.log("RECEIVING", data);
        switch (data.type) {
            case "getConversations": {
                this.conversations = data.data;
                break;
            }
            case "getFeature": {
                const feature = data.data;
                
                if(feature.data === null) {
                    this.nextBinaryFrameBelongsTo = feature;
                } else {
                    this.features.set(feature.name, feature);
                    this.gui.onFeatureReceived(feature);
                }
                break;
            }
            case "done": {
                if(this.gui.onFeatureReceiveDone) this.gui.onFeatureReceiveDone();
                break;
            }
            default: throw Error("unknown message "+(data as any).type)
        }
    }
}