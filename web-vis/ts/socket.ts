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
}

export class SocketManager {
    socket: WebSocket;
    features = new Map<string, c.Feature>();
    @observable
    conversations = [] as string[];

    constructor(private gui: c.GUI, server: string) {
        this.socket = new WebSocket(server);
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

    onSocketOpen(event: MessageEvent) {
        this.sendMessage({type: "getConversations"});
        this.loadConversation(this.gui.conversation);
    }

    @action onSocketMessage(event: MessageEvent) {
        const data: ServerMessage = JSON.parse(event.data);
        console.log("RECEIVING", data);
        switch (data.type) {
            case "getConversations": {
                this.conversations = data.data;
                break;
            }
            case "getFeature": {
                const feature = data.data;
                this.features.set(feature.name, feature);
                this.gui.onFeatureReceived(feature);
                break;
            }
            default: throw Error("unknown message "+(data as any).type)
        }
    }
}