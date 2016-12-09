import * as React from "react";
import * as ReactDOM from "react-dom";
import * as mobx from "mobx";
mobx.useStrict(true);
import { observer } from "mobx-react";
import * as util from "./util";
import DevTools from "mobx-react-devtools";
import * as s from "./socket";
import * as LZString from "lz-string";
import { autobind } from "core-decorators";
import * as B from "@blueprintjs/core";
import * as Data from "./Data";
import * as v from "./Visualizer";
import { Feature, NumFeature, NumFeatureSVector, FeatureID, ConversationID } from "./features";
import { AudioPlayer, PlaybackPosition, AudioRecorder, microphoneFeature } from "./Audio";

export const globalConfig = mobx.observable({
    maxColor: "#3232C8",
    rmsColor: "#6464DC",
    leftBarSize: 200,
    zoomFactor: 1.2,
    emptyVisHeight: 50,
    defaultConversation: "sw2807",
    minRenderDelayMS: 50
});
export class Styles {
    @mobx.computed static get leftBarCSS() {
        return { flexBasis: "content", flexGrow: 0, flexShrink: 0, width: globalConfig.leftBarSize + "px", border: "1px solid", marginRight: "5px" };
    }
    static absoluteTopRight = { position: "absolute", top: 0, right: 0 };
    static absoluteBottomRight = { position: "absolute", bottom: 0, right: 0 };
    static absoluteTopLeft = { position: "absolute", top: 0, left: 0 };
    static absoluteBottomLeft = { position: "absolute", bottom: 0, left: 0 };
}

interface Zoom {
    left: number; right: number;
}
export interface SingleUIState {
    uuid: number;
    visualizer: v.VisualizerChoice;
    feature: FeatureID;
    config: v.VisualizerConfig;
    currentRange: { min: number, max: number } | null;
}
export interface UIState {
    uuid: number;
    height: number | "auto";
    features: SingleUIState[];
}
let uuid = 0;

export function isNumFeature(f: Feature): f is NumFeature {
    return f.typ === "FeatureType.SVector" || f.typ === "FeatureType.FMatrix";
}
export const loadingSpan = <span>Loading...</span>;

class OptimizedFeaturesTree {
    getFeaturesTree(parentPath: string, category: s.CategoryTreeElement): B.ITreeNode {
        if (!category) return { id: "unused", label: "unused", childNodes: [] };

        if (typeof category === "string") {
            return {
                id: parentPath + "/" + category, label: category
            };
        }
        const path = parentPath + "/" + category.name;
        let children = category.children.map(c => this.getFeaturesTree(path, c));
        return {
            id: path,
            label: category.name,
            get childNodes(this: B.ITreeNode) {
                if (!this.isExpanded) return [];
                else return children;
            }
        };
    }
}


@observer
class ConversationSelector extends React.Component<{ gui: GUI }, {}> {
    @autobind @mobx.action
    setConversation(e: React.SyntheticEvent<HTMLInputElement>) { this.props.gui.conversationSelectorText = e.currentTarget.value; }
    async randomConversation(t?: string) {
        const conversation = await this.props.gui.randomConversation(t) as any as string;
        this.props.gui.loadConversation(conversation);
    }
    render() {
        const convos = this.props.gui.getConversations().data;
        const allconvos = convos && Object.keys(convos).map(k => convos[k]).reduce(util.mergeArray);
        return (<div style={{ display: "inline-block" }}>
            <input list="conversations" value={this.props.gui.conversationSelectorText} onChange={this.setConversation} />
            {allconvos && <datalist id="conversations">{allconvos.map(c => <option key={c as any} value={c as any} />)}</datalist>}
            <button onClick={c => this.props.gui.loadConversation(this.props.gui.conversationSelectorText)}>Load</button>
            <button onClick={() => this.randomConversation()}>Random</button>
            {convos && Object.keys(convos).map(name =>
                <button key={name} onClick={() => this.randomConversation(name)}>Random {name}</button>
            )}
        </div>);
    }
}

const examples: { [name: string]: any } = {
    "NN output": {
        uis: [
            { "features": [{ "feature": "/A/extracted/adc", "visualizer": "Waveform", "config": "givenRange", "currentRange": { "min": -32768, "max": 32768 } }], "height": "auto" },
            { "features": [{ "feature": "/A/transcript/Original/text", "visualizer": "Text", "config": "normalizeLocal", "currentRange": null }], "height": "auto" },
            { "features": [{ "feature": "/A/NN outputs/latest/best", "visualizer": "Waveform", "config": "normalizeLocal", "currentRange": { "min": 0.169875830411911, "max": 0.8434500098228455 } }], "height": "auto" },
            { "features": [{ "feature": "/A/NN outputs/latest/best.smooth", "visualizer": "Waveform", "config": "normalizeLocal", "currentRange": { "min": 0.2547765076160431, "max": 0.7286926507949829 }, "uuid": 26 }, { "feature": "/A/NN outputs/latest/best.smooth.thres", "uuid": 31, "visualizer": "Highlights", "config": "normalizeLocal", "currentRange": null }], "height": 85, "uuid": 25 },
            { "features": [{ "feature": "/A/NN outputs/latest/best.smooth.bc", "visualizer": "Waveform", "config": "normalizeLocal", "currentRange": { "min": -32768, "max": 32768 } }], "height": "auto" },
            { "features": [{ "feature": "/A/extracted/pitch", "visualizer": "Waveform", "config": "normalizeLocal", "currentRange": { "min": -1, "max": 1 } }], "height": "auto" },
            { "features": [{ "feature": "/A/extracted/power", "visualizer": "Waveform", "config": "normalizeLocal", "currentRange": { "min": -1, "max": 1 } }], "height": "auto" }]
    }
};
@observer
class MaybeAudioPlayer extends React.Component<{ gui: GUI }, {}> {
    render() {
        const gui = this.props.gui;
        if (gui.loadingState !== 1) return <span />;
        const visibleFeatures = new Set(gui.uis.map(ui => ui.features).reduce((a, b) => (a.push(...b), a), []));
        const visibleAudioFeatures = [...visibleFeatures]
            .map(f => gui.getFeature(f.feature).data)
            .filter(f => f && f.typ === "FeatureType.SVector") as NumFeatureSVector[];
        if (visibleAudioFeatures.length > 0)
            return <AudioPlayer features={visibleAudioFeatures} gui={gui} ref={gui.setAudioPlayer} />;
        else return <span />;
    }
}
@observer
class MaybeAudioRecorder extends React.Component<{ gui: GUI }, {}> {
    render() {
        const gui = this.props.gui;
        if (gui.loadingState !== 1) return <span />;
        const visibleFeatures = [...new Set(gui.uis.map(ui => ui.features).reduce((a, b) => (a.push(...b), a), []))];
        const visible = visibleFeatures.some(f => f.feature === microphoneFeature.id);
        if (visible)
            return <AudioRecorder gui={gui} />;
        else return <span />;
    }
}
@observer
export class GUI extends React.Component<{}, {}> {

    @mobx.observable windowWidth = window.innerWidth;
    @mobx.observable playbackPosition = 0;
    @mobx.observable followPlayback = false;

    @mobx.observable conversation: ConversationID;
    @mobx.observable conversationSelectorText = "";
    @mobx.observable uis = [] as UIState[];
    @mobx.observable zoom = {
        left: 0, right: 1
    };
    @mobx.observable totalTimeSeconds = NaN;

    audioPlayer: AudioPlayer; setAudioPlayer = (a: AudioPlayer) => this.audioPlayer = a;
    uisDiv: HTMLDivElement; setUisDiv = (e: HTMLDivElement) => this.uisDiv = e;
    @mobx.observable widthCalcDiv: HTMLDivElement; setWidthCalcDiv = mobx.action("setWidthCalcDiv", (e: HTMLDivElement) => this.widthCalcDiv = e);
    private socketManager: s.SocketManager;
    stateAfterLoading = null as any | null;
    @mobx.observable
    loadingState = 1;
    loadedFeatures = new Set<NumFeature>();
    @mobx.computed get categoryTree() {
        const data = this.getFeatures().data;
        if (!data) return [];
        else {
            const ft = new OptimizedFeaturesTree();
            return data.categories.map(c => ft.getFeaturesTree("", c));
        }
    }
    serialize() {
        return LZString.compressToEncodedURIComponent(JSON.stringify(mobx.toJS({
            playbackPosition: this.playbackPosition,
            followPlayback: this.followPlayback,
            conversation: this.conversation,
            uis: this.uis, // TODO: remove uuids
            zoom: this.zoom,
            totalTimeSeconds: this.totalTimeSeconds
        })));
    }
    @mobx.action
    deserialize(data: string | GUI) {
        if (this.audioPlayer) this.audioPlayer.playing = false;
        if (this.loadingState !== 1) {
            console.error("can't load while loading");
            return;
        }
        let obj;
        if (typeof data === "string") obj = JSON.parse(LZString.decompressFromEncodedURIComponent(data));
        else obj = data;
        if (obj.conversation && this.conversation !== obj.conversation) {
            this.loadConversation(obj.conversation, obj);
        } else {
            this.applyState(obj);
        }
    }
    @mobx.action
    applyState(targetState: GUI) {
        if (targetState.uis) targetState.uis.forEach(ui => { ui.uuid = uuid++; ui.features.forEach(ui => ui.uuid = uuid++); });
        Object.assign(this, targetState);
    }
    @mobx.computed
    get left() {
        this.windowWidth;
        return this.widthCalcDiv ? this.widthCalcDiv.getBoundingClientRect().left : 0;
    }
    @mobx.computed
    get width() {
        this.windowWidth;
        return this.widthCalcDiv ? this.widthCalcDiv.clientWidth : 100;
    }
    @mobx.action
    async loadConversation(conversation: string, targetState?: GUI) {
        const convID = await this.verifyConversationID(conversation);
        mobx.runInAction("resetUIs", () => {
            this.uis = [];
            this.conversation = convID;
            this.zoom.left = 0; this.zoom.right = 1;
            this.totalTimeSeconds = NaN;
            this.loadedFeatures.clear();
            this.loadingState = 0;
            this.conversationSelectorText = conversation;
        });
        const features = await this.getFeatures();
        const targetFeatures = targetState ? targetState.uis.map(ui => ui.features.map(ui => ui.feature)) : features.defaults;
        const total = targetFeatures.reduce((sum, next) => sum + next.length, 0);
        let i = 0;
        for (const featureIDs of targetFeatures) {
            const ui = this.getDefaultUIState([]);
            mobx.runInAction("addDefaultUI", () => {
                this.uis.push(ui);
            });
            for (const featureID of featureIDs) {
                const feat = await this.getFeature(featureID);
                mobx.runInAction("progressIncrement", () => {
                    ui.features.push(this.getDefaultSingleUIState(feat));
                    this.loadingState = ++i / total;
                });
            }
        }
        if (targetState) {
            this.applyState(targetState);
        }
    }
    async verifyConversationID(id: string): Promise<ConversationID> {
        const convos = await this.socketManager.getConversations();
        if (Object.keys(convos).some(name => convos[name].indexOf(id as any) >= 0)) return id as any;
        throw Error("unknown conversation " + id);
    }
    async randomConversation(category?: string): Promise<ConversationID> {
        const convos = await this.getConversations();
        let choices;
        if (!category) choices = Object.keys(convos).map(k => convos[k]).reduce(util.mergeArray);
        else choices = convos[category];
        return util.randomChoice(choices);
    }
    @mobx.action
    onWheel(event: MouseWheelEvent) {
        if (!event.ctrlKey) return;
        event.preventDefault();
        const position = util.getPositionFromPixel(event.clientX, this.left, this.width, this.zoom) !;
        const scaleChange = event.deltaY > 0 ? globalConfig.zoomFactor : 1 / globalConfig.zoomFactor;
        this.zoom = util.rescale(this.zoom, scaleChange, position);
        this.zoom.right = Math.min(this.zoom.right, 1);
        this.zoom.left = Math.max(this.zoom.left, 0);
    }
    getDefaultSingleUIState(feature: Feature): SingleUIState {
        let visualizerConfig: v.VisualizerConfig;
        if (isNumFeature(feature) && feature.range instanceof Array) {
            visualizerConfig = "givenRange";
        } else visualizerConfig = "normalizeLocal";
        return {
            feature: feature.name,
            uuid: uuid++,
            visualizer: v.getVisualizerChoices(feature)[0],
            config: visualizerConfig,
            currentRange: mobx.asStructure(null),
        };
    }
    getDefaultUIState(features: Feature[]): UIState {
        return {
            uuid: uuid++,
            features: features.map(f => this.getDefaultSingleUIState(f)),
            height: "auto"
        };
    }
    checkTotalTime(feature: Feature) {
        if (isNumFeature(feature)) {
            if (this.loadedFeatures.has(feature)) return;
            let totalTime: number = NaN;
            if (feature.typ === "FeatureType.SVector") {
                totalTime = feature.data.shape[0] / (feature.samplingRate * 1000);
            } else if (feature.typ === "FeatureType.FMatrix") {
                totalTime = feature.data.shape[0] * feature.shift / 1000;
            }
            if (!isNaN(totalTime)) {
                if (isNaN(this.totalTimeSeconds)) mobx.runInAction("setTotalTime", () => this.totalTimeSeconds = totalTime);
                else if (Math.abs((this.totalTimeSeconds - totalTime) / totalTime) > 0.001) {
                    console.error("Mismatching times, was ", this.totalTimeSeconds, "but", feature.name, "has length", totalTime);
                }
            }
        }
    }
    @autobind
    onSocketOpen() {
        if (location.hash.length > 1) {
            this.deserialize(location.hash.substr(1));
        } else {
            this.loadConversation(globalConfig.defaultConversation);
        }
    }
    @autobind
    windowResize() {
        if (this.windowWidth !== document.body.clientWidth)
            mobx.runInAction("windowWidthChange", () => this.windowWidth = document.body.clientWidth);
    }
    constructor() {
        super();
        this.socketManager = new s.SocketManager(`ws://${location.host.split(":")[0]}:8765`);
        window.addEventListener("wheel", e => this.onWheel(e));
        window.addEventListener("resize", this.windowResize);
        // window resize is not fired when scroll bar appears. wtf.
        setInterval(this.windowResize, 200);
        window.addEventListener("hashchange", mobx.action("hashChange", e => this.deserialize(location.hash.substr(1))));
        this.socketManager.socketOpen().then(this.onSocketOpen);
    }
    getFeature(id: string | FeatureID) {
        if (id === microphoneFeature.id) {
            return new s.LulPromise(AudioRecorder.getFeature());
        }
        const f = this.socketManager.getFeature(this.conversation, id as any as FeatureID);
        f.then(f => this.checkTotalTime(f));
        return f;
    }
    getFeatures() {
        return this.socketManager.getFeatures(this.conversation).then(features => {
            return {
                ...features,
                categories: [...features.categories, microphoneFeature.name],
            };
        });
    }
    getConversations() {
        return this.socketManager.getConversations();
    }
    @mobx.action
    addUI(afterUuid: number) {
        this.uis.splice(this.uis.findIndex(ui => ui.uuid === afterUuid) + 1, 0, { uuid: uuid++, features: [], height: "auto" });
    }
    *getVisualizers() {
        for (const ui of this.uis) {
            yield <v.InfoVisualizer key={ui.uuid} uiState={ui} gui={this} />;
            yield <button key={ui.uuid + "+"} onClick={() => this.addUI(ui.uuid)}>Add Visualizer</button>;
        }
    }
    render(): JSX.Element {
        const self = this;
        const ProgressIndicator = observer(function ProgressIndicator() {
            if (self.loadingState === 1) return <span>Loading complete</span>;
            return (
                <div style={{ display: "inline-block", width: "200px" }}>
                    Loading: <B.ProgressBar intent={B.Intent.PRIMARY} value={self.loadingState} />
                </div>
            );
        });
        return (
            <div>
                <div style={{ margin: "10px" }} className="headerBar">
                    <ConversationSelector gui={this} />
                    <label>Follow playback:
                        <input type="checkbox" checked={this.followPlayback}
                            onChange={mobx.action("changeFollowPlayback", (e: React.SyntheticEvent<HTMLInputElement>) => this.followPlayback = e.currentTarget.checked)} />
                    </label>
                    <span>Playback position: <PlaybackPosition gui={this} /></span>
                    <MaybeAudioRecorder gui={this} />
                    <button onClick={() => location.hash = "#" + this.serialize()}>Serialize → URL</button>
                    <ProgressIndicator />
                    {Object.keys(examples).length > 0 && <span>Examples: {Object.keys(examples).map(k =>
                        <button key={k} onClick={e => this.deserialize(examples[k])}>{k}</button>)}</span>
                    }
                </div>
                <div ref={this.setUisDiv}>
                    <div style={{ display: "flex", visibility: "hidden" }}>
                        <div style={Styles.leftBarCSS} />
                        <div style={{ flexGrow: 1 }} ref={this.setWidthCalcDiv} />
                    </div>
                    {[...this.getVisualizers()]}
                </div>
                <MaybeAudioPlayer gui={this} />
                <DevTools />
            </div>
        );
    }
}



const _gui = ReactDOM.render(<GUI />, document.getElementById("root")) as GUI;

Object.assign(window, { gui: _gui, util, globalConfig, mobx, Data });