type ServerMessages = {
}
export type BCSamples = {name: string, samples: string[]};
type ClientRPCs = {
    getData: {
        request: {},
        response: {
            sessionId: number,
            monosegs: string[],
            bcSamples: BCSamples[],
            netRatingSegments: string[]
        }
    },
    beginStudy: {
        request: {bcSampleSource: string},
        response: {}
    },
    submitBC: {
        request: {time: number, duration: number, segment: string},
        response: {}
    }
    submitNetRatings: {
        request: {segments: [string, number][], final: boolean},
        response: {}
    }
    comment: {
        request: string,
        response: {}
    }
}

type ClientMessages = {
}

export interface RouletteClientSocket {
    on<K extends keyof ServerMessages>(type: K, listener: (info: ServerMessages[K]) => void): this;

    emit<K extends keyof ClientMessages>(type: K, info: ClientMessages[K]): this;

    emit<K extends keyof ClientRPCs>(type: K, info: ClientRPCs[K]["request"], callback: (data: ClientRPCs[K]["response"]) => void): this;
    disconnect(): any;
}

export interface RouletteServerSocket {
    on<K extends keyof ClientMessages>(type: K, listener: (info: ClientMessages[K]) => void): this;
    on<K extends keyof ClientRPCs>(type: K, listener: (info: ClientRPCs[K]["request"], callback: (data: ClientRPCs[K]["response"]) => void) => void): this;

    emit<K extends keyof ServerMessages>(type: K, info: ServerMessages[K]): this;
}
