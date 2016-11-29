import * as React from 'react';
import * as ReactDOM from 'react-dom';
import { autorun, computed, observable, action, extendObservable, isObservableArray, asMap, toJS, runInAction, asStructure } from 'mobx';
import * as mobx from 'mobx';
mobx.useStrict(true);
import { observer } from 'mobx-react';
import * as Waveform from './Waveform';
import * as util from './util';
import DevTools from 'mobx-react-devtools';
import * as s from './socket';
import * as LZString from 'lz-string';
import * as highlights from './Highlights';
import {autobind} from 'core-decorators';
import * as B from '@blueprintjs/core';
import * as Data from './Data';
export const globalConfig = observable({
    maxColor: "#3232C8",
    rmsColor: "#6464DC",
    leftBarSize: 200,
    zoomFactor: 1.2,
    emptyVisHeight: 50,
    defaultConversation: "sw2807",
    minRenderDelayMS: 50
});
export class styles {
    @computed static get leftBarCSS() {
        return {flexBasis: "content", flexGrow:0, flexShrink: 0, width:globalConfig.leftBarSize+"px", border:"1px solid", marginRight:"5px"}
    }
}

export interface VisualizerProps<T> {
    gui: GUI;
    uiState: SingleUIState;
    feature: T;
}
export abstract class Visualizer<T> extends React.Component<VisualizerProps<T>, {}> {
    preferredHeight: number;
}
export interface VisualizerConstructor<T> {
    new (props?: VisualizerProps<T>, context?: any): Visualizer<T>;
}
export type NumFeatureCommon = {
    name: string,
    samplingRate: number, // in kHz
    shift: number,
    data: Data.TwoDimensionalArray,
    range: [number, number] | null
};
export type NumFeatureSVector = NumFeatureCommon & {
    typ: "FeatureType.SVector", dtype: "int16"
};

export type NumFeatureFMatrix = NumFeatureCommon & {
    typ: "FeatureType.FMatrix", dtype: "float32"
};
export type Color = [number, number, number];

export type Utterances = {
    name: string,
    typ: "utterances",
    data: Utterance[]
}
export type Highlights = {
    name: string,
    typ: "highlights",
    data: Utterance[]
}
export type NumFeature = NumFeatureSVector | NumFeatureFMatrix;
export type Feature = NumFeature | Utterances | Highlights;
export type Utterance = {from: number | string, to: number | string, text?: string, id?: string, color?: Color};
export type VisualizerConfig  = "normalizeGlobal" | "normalizeLocal" | "givenRange";

export interface SingleUIState {
    visualizer: VisualizerChoice;
    feature: string;
    config: VisualizerConfig;
    currentRange: {min: number, max: number} | null;
}
export interface UIState {
    uuid: number,
    height: number | "auto",
    features: SingleUIState[];
}
let uuid = 0;
interface Zoom {
    left: number; right: number;
}
export function isNumFeature(f: Feature): f is NumFeature {
    return f.typ === "FeatureType.SVector" || f.typ === "FeatureType.FMatrix";
}
export const loadingSpan = <span>Loading...</span>;

class OptimizedFeaturesTree {
    getFeaturesTree(parentPath: string, category: s.CategoryTreeElement): B.ITreeNode {
        if(!category) return {id: "unused", label: "unused", childNodes: []};

        if(s.isFeatureID(category)) {
            const name = category as any as string;
            return {
                id: parentPath + "/" + name, label: name
            }
        }
        const path = parentPath + "/" + category.name;
        let children = category.children.map(c => this.getFeaturesTree(path, c));
        return {
            id: path,
            label: category.name,
            get childNodes(this: B.ITreeNode) {
                if(!this.isExpanded) return [];
                else return children;
            }
        }
    }
}

@observer
class CategoryTree extends React.Component<{gui: GUI, features: s.GetFeaturesResponse, onClick:(feat: string) => void}, {}> {
    constructor(props: any) {
        super(props);
        this.currentTree = this.props.gui.categoryTree;
    }
 
    currentTree: B.ITreeNode[];
    @autobind @action
    handleNodeClick(n: B.ITreeNode) {if(!n.childNodes) this.props.onClick(""+n.id); }
    @autobind @action
    handleNodeExpand(n: B.ITreeNode) {n.isExpanded = true;this.forceUpdate();}
    @autobind @action
    handleNodeCollapse(n: B.ITreeNode) {n.isExpanded = false;this.forceUpdate();}
    render() {
        return <div>
            <B.Tree contents={this.currentTree}
                    onNodeCollapse={this.handleNodeCollapse}
                    onNodeClick={this.handleNodeClick}
                    onNodeExpand={this.handleNodeExpand} />
            <button className="pt-button pt-popover-dismiss">Cancel</button>
        </div>;
    }
}
@observer
class LeftBar extends React.Component<{uiState: UIState, gui: GUI}, {}> {
    static rangeOptions = ["normalizeGlobal", "normalizeLocal", "givenRange"]
    @action changeVisualizerConfig(info: SingleUIState, value: string) {
        info.config = value as VisualizerConfig;
    }
    @action changeVisualizer(info: SingleUIState, value: string) {
        info.visualizer = value as VisualizerChoice;
    }
    async changeFeature(e: React.SyntheticEvent<HTMLSelectElement>, i: number) {
        const state = this.props.gui.getDefaultSingleUIState(await this.props.gui.getFeature(e.currentTarget.value).promise)
        runInAction("changeFeature"+i, () => this.props.uiState.features[i] = state);
    }
    @action remove(i: number) {
        this.props.uiState.features.splice(i, 1);
        if(this.props.uiState.features.length === 0) {
            const uis = this.props.gui.uis;
            uis.splice(uis.findIndex(ui => ui.uuid === this.props.uiState.uuid), 1);
        }
    }
    @action async add(feat: string) {
        this.addPopover.setState({isOpen: false});
        const gui = this.props.gui;
        const state = gui.getDefaultSingleUIState(await gui.getFeature(feat).promise);
        runInAction(() => this.props.uiState.features.push(state));
    }
    addPopover: B.Popover;
    render() {
        let minmax;
        const {uiState, gui} = this.props;
        const firstWithRange = uiState.features.find(props => props.currentRange !== null);
        if(firstWithRange && firstWithRange.currentRange) {
            minmax = [
                <div key="max" style={{position: "absolute", margin:0, top:0, right:0}}>{util.round1(firstWithRange.currentRange.max)}</div>,
                <div key="min" style={{position: "absolute", margin:0, bottom:0, right:0}}>{util.round1(firstWithRange.currentRange.min)}</div>,
            ];
        } else minmax = "";
        const VisualizerChoices = observer((props:{info: SingleUIState}) => {
            const feature = gui.getFeature(props.info.feature).data;
            if(!feature) return <span/>;
            const c = getVisualizerChoices(feature);
            if(c.length > 1) return (
                <label className="pt-label pt-inline">Visualizer
                    <div className="pt-select">
                        <select value={props.info.visualizer} onChange={e => this.changeVisualizer(props.info, e.currentTarget.value)}>
                            {c.map(c => <option key={c} value={c}>{c}</option>)}
                        </select>
                    </div>
                </label>
            );
            else return <span/>;
        });
        const features = gui.getFeatures().data;
        return (
            <div className="left-bar" style={{position: "relative", width:"100%", height:"100%"}}>
                <div style={{position: "absolute", top:0, bottom:0, left:0, paddingLeft:"5px", display:"flex", justifyContent:"center", flexDirection:"column",alignItems:"flex-start"}}>
                    {uiState.features.map((info, i) => 
                        <B.Popover key={i} interactionKind={B.PopoverInteractionKind.HOVER} popoverClassName="change-visualizer"
                            content={<div>
                                    <label className="pt-label pt-inline"><button className="pt-button pt-intent-danger pt-icon-remove" onClick={e => this.remove(i)}>Remove</button></label>
                                    {info.currentRange&&
                                        <label className="pt-label pt-inline">Range
                                            <div className="pt-select">
                                                <select value={info.config} onChange={e => this.changeVisualizerConfig(info, e.currentTarget.value)}>
                                                    {LeftBar.rangeOptions.map(op => <option key={op} value={op}>{op}</option>)}
                                                </select>
                                            </div>
                                        </label>
                                    }
                                <VisualizerChoices info={info} />
                                </div>
                            }><div className="pt-tooltip-indicator" style={{ marginBottom:"5px"}}>{info.feature}</div></B.Popover>
                        
                    )}
                </div>
                <div style={{position:"absolute", bottom:0, left:0, margin:"5px"}}>
                    <B.Popover content={features ? <CategoryTree gui={gui} features={features} onClick={e => this.add(e)} />: "not loaded" }
                            interactionKind={B.PopoverInteractionKind.CLICK}
                            position={B.Position.RIGHT}
                            popoverClassName="add-visualizer"
                            useSmartPositioning={true} ref={p => this.addPopover = p}>
                            <button>Add feature</button>
                    </B.Popover>
                </div>
                {minmax}
            </div>
        );
    }
}
@observer
class InfoVisualizer extends React.Component<{uiState: UIState, gui: GUI}, {}> {
    render() {
        const {uiState, gui} = this.props;
        
        return (
            <div style={{display: "flex", boxShadow: "-9px 10px 35px -10px rgba(0,0,0,0.29)",zIndex: 1}}>
                <div style={styles.leftBarCSS}>
                    <LeftBar gui={gui} uiState={uiState} />
                </div>
                <div style={{flexGrow: 1}}>
                    <highlights.OverlayVisualizer gui={gui} uiState={uiState} />
                </div>
            </div>
        );

    }
}
@observer
class TextVisualizer extends Visualizer<Utterances> {
    @observable
    tooltip: number|null = null;
    @computed get playbackTooltip() {
        const data = this.props.feature.data;
        const b = util.binarySearch(0, data.length, x => +data[x].from, this.props.gui.playbackPosition * this.props.gui.totalTimeSeconds);
        return b;
    }
    // @computed currentlyVisibleTh
    getElements() {
        const width = this.props.gui.width;
        return this.props.feature.data.map((utt,i) => {
            const from = +utt.from / this.props.gui.totalTimeSeconds, to = +utt.to / this.props.gui.totalTimeSeconds;
            let left = util.getPixelFromPosition(from, 0, width, this.props.gui.zoom);
            let right = util.getPixelFromPosition(to, 0, width, this.props.gui.zoom);
            if ( right < 0 || left > this.props.gui.width) return null;
            const style = {};
            if(utt.color) Object.assign(style, {backgroundColor: `rgb(${utt.color})`});
            let className = "utterance utterance-text";
            if(left < 0) {
                left = 0;
                Object.assign(style, {borderLeft: "none"});
                className += " leftcutoff";
            }
            if(right > width) {
                right = width;
                Object.assign(style, {borderRight: "none"});
                className += " rightcutoff";
            }
            const padding = 3;
            Object.assign(style, {left:left+"px", width: (right-left)+"px", padding: padding +"px"});
            return <div className={className} key={utt.id !== undefined ? utt.id : i} style={style}
                    onMouseEnter={action("hoverTooltip", _ => this.tooltip = i)}
                    onMouseLeave={action("hoverTooltipDisable", _ => this.tooltip = null)}>
                <span>{utt.text}</span>
            </div>;
        });
    }
    getTooltip(i: number) {
        const width = this.props.gui.width;
        const utt = this.props.feature.data[i];
        const from = +utt.from / this.props.gui.totalTimeSeconds, to = +utt.to / this.props.gui.totalTimeSeconds;
        let left = util.getPixelFromPosition(from, 0, width, this.props.gui.zoom);
        let right = util.getPixelFromPosition(to, 0, width, this.props.gui.zoom);
        if ( right < 0 || left > this.props.gui.width) return null;
        let className = "utterance tooltip visible";
        let styleText;
        if(utt.color) styleText = {backgroundColor: `rgb(${utt.color})`}
        else styleText = {};
        const style = {left:left+"px", width: (right-left)+"px"};
        return <div className={className} key={utt.id} style={style}>
            <span className="content" style={styleText}><b/>{utt.text}</span>
        </div>;
    }
    Tooltip = observer(function Tooltip(this: TextVisualizer) {
        return <div>
            {this.playbackTooltip !== null && this.props.gui.audioPlayer && this.props.gui.audioPlayer.playing &&
                <div style={{position: "relative", height: "0px", width:"100%"}}>{this.getTooltip(this.playbackTooltip)}</div>}
            {this.tooltip !== null && <div style={{position: "relative", height: "0px", width:"100%"}}>{this.getTooltip(this.tooltip)}</div>}
        </div>;
    }.bind(this))
    render() {
        return (
            <div style={{height: "4em"}}>
                <div style={{overflow: "hidden", position: "relative", height: "40px", width:"100%"}}>{this.getElements()}</div>
                <this.Tooltip />
            </div>
        );
    }
}

class RobinAudioNode {
    constructor(private buf: AudioBuffer) {

    }
}
@observer
class AudioPlayer extends React.Component<{features: NumFeatureSVector[], zoom: Zoom, gui: GUI}, {}> {
    playerBar: HTMLDivElement; setPlayerBar = (p: HTMLDivElement) => this.playerBar = p;
    disposers: (() => void)[] = [];
    audio: AudioContext;
    @observable
    playing: boolean;
    startedAt: number;
    duration: number;
    startPlayingAtom = new mobx.Atom("Player");
    constructor(props: any) {
        super(props);
        this.audio = new AudioContext();
        this.disposers.push(() => (this.audio as any).close());
        this.disposers.push(mobx.autorun(() => {
            for(const feature of this.props.features) this.makeAudioBuffer(feature);
            if(this.playing) {
                for(const feature of this.props.features) {
                    const buffer = this.makeAudioBuffer(feature);
                    const audioSource = this.toAudioSource({buffer, audio: this.audio});
                
                    this.duration = buffer.duration;
                    audioSource.playbackRate.value = 1;
                    const startPlaybackPosition = mobx.untracked(() => this.props.gui.playbackPosition);
                    audioSource.start(0, startPlaybackPosition * buffer.duration);
                    this.startedAt = this.audio.currentTime - startPlaybackPosition * buffer.duration;
                    audioSource.addEventListener("ended", action("audioEnded", () => this.playing = false));
                }
                requestAnimationFrame(this.updatePlaybackPosition);
            }
        }));
    }

    center() {
        const zoom = this.props.gui.zoom;
        const w = zoom.right - zoom.left;
        let pos = this.props.gui.playbackPosition;
        if (pos - w/2 < 0) pos = w/2;
        if (pos + w/2 > 1) pos = 1 - w/2;
        zoom.left = pos - w/2; zoom.right = pos + w/2;
    }
    @autobind @action
    updatePlaybackPosition() {
        if(!this.playing) return;
        this.props.gui.playbackPosition = (this.audio.currentTime - this.startedAt) / this.duration;
        if(this.props.gui.followPlayback) this.center();
        if(this.playing) requestAnimationFrame(this.updatePlaybackPosition);
    }
    @computed get xTranslation() {
        const x = util.getPixelFromPosition(this.props.gui.playbackPosition, this.props.gui.left, this.props.gui.width, this.props.zoom);
        return "translateX("+x+"px)";
    }
    makeAudioBuffer = mobx.createTransformer((feature: NumFeature) => {
        console.log("creating buffer for " + feature.name);
        const audioBuffer = this.audio.createBuffer(1, feature.data.shape[0], feature.samplingRate * 1000);
        feature.data.iterate("ALL", 0);
        const arr = Float32Array.from(feature.data.buffer, v => v / 2 ** 15);
        audioBuffer.copyToChannel(arr, 0);
        return audioBuffer;
    });
    toAudioSource = mobx.createTransformer(({buffer, audio}: {buffer: AudioBuffer, audio: AudioContext}) => {
        const audioSource = audio.createBufferSource();
        audioSource.buffer = buffer;
        audioSource.playbackRate.value = 0;
        audioSource.connect(audio.destination);
        return audioSource;
    }, buf => buf.stop());

    @autobind @action
    onKeyUp(event: KeyboardEvent) {
        if(event.keyCode == 32) {
            event.preventDefault();
            this.playing = !this.playing;
        }
    }
    @autobind @action
    onKeyDown(event: KeyboardEvent) {
        if(event.keyCode == 32) {
            event.preventDefault();
        }
    }
    @autobind @action
    onClick(event: MouseEvent) {
        if(event.clientX < this.props.gui.left) return;
        event.preventDefault();
        const x = util.getPositionFromPixel(event.clientX, this.props.gui.left, this.props.gui.width, this.props.zoom)!;
        this.playing = false;
        runInAction("clickSetPlaybackPosition", () => this.props.gui.playbackPosition = Math.max(x, 0));
    }

    componentDidMount() {
        const {gui, zoom} = this.props;
        const uisDiv = this.props.gui.uisDiv;
        uisDiv.addEventListener("click", this.onClick);
        window.addEventListener("keydown", this.onKeyDown);
        window.addEventListener("keyup", this.onKeyUp);
        this.disposers.push(() => uisDiv.removeEventListener("click", this.onClick));
        this.disposers.push(() => window.removeEventListener("keydown", this.onKeyDown));
        this.disposers.push(() => window.removeEventListener("keyup", this.onKeyUp));
        this.disposers.push(action("stopPlaybackExit", () => this.playing = false));
    }
    componentWillUnmount() {
        for(const disposer of this.disposers) disposer();
    }
    render() {
        return (
            <div ref={this.setPlayerBar} style={{position: "fixed", width: "2px", height: "100vh", top:0, left:0, transform: this.xTranslation, backgroundColor:"gray"}} />
        );
    }
}
type VisualizerChoice = "Waveform"|"Darkness"|"Text"|"Highlights";

export function getVisualizerChoices(feature: Feature): VisualizerChoice[] {
    if(!feature) return [];
    if(feature.typ === "FeatureType.SVector" || feature.typ === "FeatureType.FMatrix") {
        return ["Waveform", "Darkness"];
    } else if (feature.typ === "utterances") {
        return ["Text", "Highlights"];
    } else if (feature.typ === "highlights") {
        return ["Highlights", "Text"];
    }else throw Error("Can't visualize " + (feature as any).typ);
}

@observer
export class ChosenVisualizer extends React.Component<VisualizerProps<Feature>,{}> {
    @observable
    preferredHeight: number;
    static visualizers:{[name: string]: VisualizerConstructor<any>} = {
        "Waveform": Waveform.AudioWaveform,
        "Darkness": Waveform.Darkness,
        "Text": TextVisualizer,
        "Highlights": highlights.HighlightsVisualizer
    }
    render() {
        const Visualizer = ChosenVisualizer.visualizers[this.props.uiState.visualizer];
        return <Visualizer {...this.props} ref={action((e: Visualizer<any>) => this.preferredHeight = e?e.preferredHeight:0)} />;
    }
}

function splitIn(count: number, data: Utterances) {
    const feats: Utterances[] = [];
    for(let i = 0; i < count; i++) {
        feats[i] = {name: data.name+"."+i, typ: data.typ, data: []};
    }
    data.data.forEach((utt, i) => feats[i%count].data.push(utt));
    return feats;
}

const PlaybackPosition = observer(function PlaybackPosition({gui}: {gui: GUI}) {
    return <span>{(gui.playbackPosition * gui.totalTimeSeconds).toFixed(4)}</span>
});

@observer
class ConversationSelector extends React.Component<{gui: GUI}, {}> {
    @observable
    conversation = "sw2001";
    @autobind @action
    setConversation(e: React.SyntheticEvent<HTMLInputElement>) { this.conversation = e.currentTarget.value; }
    @autobind
    async randomConversation() {
        const conversation = await this.props.gui.randomConversation() as any as string;
        runInAction(() => this.conversation = conversation);
        this.props.gui.loadConversation(conversation);
    }
    render() {
        const convos = this.props.gui.getConversations().data;
        return (<div style={{display:"inline-block"}}>
            <input list="conversations" value={this.conversation} onChange={this.setConversation} />
            {convos && <datalist id="conversations">{convos.map(c => <option key={c as any} value={c as any}/>)}</datalist>}
            <button onClick={c => this.props.gui.loadConversation(this.conversation)}>Load</button>
            <button onClick={this.randomConversation}>RND</button>
        </div>)
    }
}

const badExamples: {[name: string]: string} = {
}
let MaybeAudioPlayer = observer<{gui:GUI}>(function MaybeAudioPlayer({gui}: {gui: GUI}) {
    if(gui.loadingState !== 1) return <span/>;
    const visibleFeatures = new Set(gui.uis.map(ui => ui.features).reduce((a,b) => (a.push(...b),a), []));
    const visibleAudioFeatures = [...visibleFeatures]
        .map(f => gui.getFeature(f.feature).data)
        .filter(f => f && f.typ === "FeatureType.SVector") as NumFeatureSVector[];
    if(visibleAudioFeatures.length > 0)
        return <AudioPlayer features={visibleAudioFeatures} zoom={gui.zoom} gui={gui} ref={gui.setAudioPlayer} />;
    else return <span/>;
});
@observer
export class GUI extends React.Component<{}, {}> {
    
    @observable windowWidth = window.innerWidth;
    @observable playbackPosition = 0;
    @observable followPlayback = true;

    @observable conversation: s.ConversationID;
    @observable uis = [] as UIState[];
    @observable zoom = {
        left: 0, right: 1
    };
    @observable totalTimeSeconds = NaN;

    audioPlayer: AudioPlayer; setAudioPlayer = (a: AudioPlayer) => this.audioPlayer = a;
    uisDiv: HTMLDivElement; setUisDiv = (e: HTMLDivElement) => this.uisDiv = e;
    @observable widthCalcDiv: HTMLDivElement; setWidthCalcDiv = action("setWidthCalcDiv", (e:HTMLDivElement) => this.widthCalcDiv = e);
    private socketManager: s.SocketManager;
    stateAfterLoading = null as any | null;
    @observable
    loadingState = 1;
    loadedFeatures = new Set<NumFeature>();
    @computed get categoryTree() {
        const data =  this.socketManager.getFeatures(this.conversation).data;
        if(!data) return [];
        else {
            const ft = new OptimizedFeaturesTree();
            return data.categories.map(c => ft.getFeaturesTree("", c));
        }
    }
    serialize() {
        return LZString.compressToEncodedURIComponent(JSON.stringify(toJS({
            playbackPosition: this.playbackPosition,
            followPlayback: this.followPlayback,
            conversation: this.conversation,
            uis: this.uis, //TODO: fix uuids
            zoom: this.zoom,
            totalTimeSeconds: this.totalTimeSeconds
        })));
    }
    @action
    deserialize(data: string) {
        if(this.audioPlayer) this.audioPlayer.playing = false;
        if(this.loadingState !== 1) {
            console.error("can't load while loading");
            return;
        }
        const obj = JSON.parse(LZString.decompressFromEncodedURIComponent(data));
        if(this.conversation !== obj.conversation) {
            this.stateAfterLoading = obj;
            this.loadConversation(obj.conversation);
        } else {
            Object.assign(this, obj);
        }
    }
    @computed
    get left() {
        this.windowWidth;
        return this.widthCalcDiv ? this.widthCalcDiv.getBoundingClientRect().left : 0;
    }
    @computed
    get width() {
        this.windowWidth;
        return this.widthCalcDiv ? this.widthCalcDiv.clientWidth : 100;
    }
    @action
    async loadConversation(conversation: string) {
        const convID = await this.verifyConversationID(conversation);
        runInAction("resetUIs", () => {
            this.uis = [];
            this.conversation = convID;
            this.zoom.left = 0; this.zoom.right = 1;
            this.totalTimeSeconds = NaN;
            this.loadedFeatures.clear();
            this.loadingState = 0;
        });
        const features = await this.socketManager.getFeatures(convID).promise;
        const total = features.defaults.reduce((sum, next) => sum + next.length, 0);
        let i = 0;
        for(const featureIDs of features.defaults) {
            const feats = [] as Feature[];
            const ui = this.getDefaultUIState([]);
            runInAction("addDefaultUI", () => {
                this.uis.push(ui)
            });
            for(const featureID of featureIDs) {
                const feat = await this.getFeature(featureID).promise;
                runInAction("progressIncrement", () => {
                    ui.features.push(this.getDefaultSingleUIState(feat))
                    this.loadingState = ++i / total;
                });
            }
            
        }
    }
    async verifyConversationID(id: string): Promise<s.ConversationID> {
        const convos = await this.socketManager.getConversations().promise;
        if(convos.indexOf(id as any) >= 0) return id as any;
        throw Error("unknown conversation " + id);
    }
    async randomConversation(): Promise<s.ConversationID> {
        return util.randomChoice(await this.getConversations().promise);
    }
    @action
    onWheel(event: MouseWheelEvent) {
            if (!event.ctrlKey) return;
            event.preventDefault();
            const position = util.getPositionFromPixel(event.clientX, this.left, this.width, this.zoom)!;
            const scale = 1/(this.zoom.right - this.zoom.left);
            const scaleChange = event.deltaY > 0 ? globalConfig.zoomFactor : 1/globalConfig.zoomFactor;
            const newScale = scale * scaleChange;
            this.zoom.left -= position;
            this.zoom.right -= position;
            this.zoom.left *= scaleChange; 
            this.zoom.right *= scaleChange;
            this.zoom.left += position;
            this.zoom.right += position;
            this.zoom.right = Math.min(this.zoom.right, 1);
            this.zoom.left = Math.max(this.zoom.left, 0);
    }
    getDefaultSingleUIState(feature: Feature): SingleUIState {
        let visualizerConfig: VisualizerConfig;
        if(isNumFeature(feature) && feature.range instanceof Array) {
            visualizerConfig = "givenRange";
        } else visualizerConfig = "normalizeLocal";
        return {
            feature: feature.name,
            visualizer: getVisualizerChoices(feature)[0],
            config: visualizerConfig,
            currentRange: asStructure(null),
        }
    }
    getDefaultUIState(features: Feature[]): UIState {
        return {
            uuid: uuid++,
            features: features.map(f => this.getDefaultSingleUIState(f)),
            height: "auto"
        }
    }
    checkTotalTime(feature: Feature) {
        if(isNumFeature(feature))  {
            if(this.loadedFeatures.has(feature)) return;
            let totalTime: number = NaN;
            if(feature.typ === "FeatureType.SVector") {
                totalTime = feature.data.shape[0] / (feature.samplingRate * 1000);
            } else if(feature.typ === "FeatureType.FMatrix") {
                totalTime = feature.data.shape[0] * feature.shift / 1000;
            }
            if (!isNaN(totalTime)) {
                if(isNaN(this.totalTimeSeconds)) runInAction("setTotalTime", () => this.totalTimeSeconds = totalTime);
                else if(Math.abs((this.totalTimeSeconds - totalTime) / totalTime) > 0.001) {
                    console.error("Mismatching times, was ", this.totalTimeSeconds, "but", feature.name, "has length", totalTime);
                }
            }
        }
    }
    @autobind
    onSocketOpen() {
        if(location.hash.length > 1) {
            this.deserialize(location.hash.substr(1));
        } else {
            this.loadConversation(globalConfig.defaultConversation);
        }
    }
    @autobind
    windowResize() {
        if(this.windowWidth !== document.body.clientWidth)
            runInAction("windowWidthChange", () => this.windowWidth = document.body.clientWidth);
    }
    constructor() {
        super();
        this.socketManager = new s.SocketManager(`ws://${location.host.split(":")[0]}:8765`);
        window.addEventListener("wheel", e => this.onWheel(e));
        window.addEventListener("resize", this.windowResize);
        // window resize is not fired when scroll bar appears. wtf.
        setInterval(this.windowResize, 200)
        window.addEventListener("hashchange", action("hashChange", e => this.deserialize(location.hash.substr(1))));
        this.socketManager.socketOpen().then(this.onSocketOpen);
    }
    getFeature(id: string|s.FeatureID) {
        const f = this.socketManager.getFeature(this.conversation, id as any as s.FeatureID);
        if(f.data) this.checkTotalTime(f.data);
        else f.promise.then(f => this.checkTotalTime(f));
        return f;
    }
    getFeatures() {
        return this.socketManager.getFeatures(this.conversation);
    }
    getConversations() {
        return this.socketManager.getConversations();
    }
    @action
    addUI(afterUuid: number) {
        this.uis.splice(this.uis.findIndex(ui => ui.uuid === afterUuid) + 1, 0, {uuid: uuid++, features:[], height: "auto"});
    }
    *getVisualizers() {
        for(const ui of this.uis) {
            yield <InfoVisualizer key={ui.uuid} uiState={ui} gui={this} />;
            yield <button key={ui.uuid+"+"} onClick={() => this.addUI(ui.uuid)}>Add Visualizer</button>;
        }
    }
    render(): JSX.Element {
        const self = this;
        const ProgressIndicator = observer(function ProgressIndicator() {
            if(self.loadingState === 1) return <span>Loading complete</span>;
            return (
                <div style={{display:"inline-block", width:"200px"}}>
                    Loading: <B.ProgressBar intent={B.Intent.PRIMARY} value={self.loadingState} />
                </div>
            );
        });
        return (
            <div>
                <div style={{margin: "10px"}} className="headerBar">
                    <ConversationSelector gui={this} />
                    <label>Follow playback:
                        <input type="checkbox" checked={this.followPlayback}
                            onChange={action("changeFollowPlayback", (e: React.SyntheticEvent<HTMLInputElement>) => this.followPlayback = e.currentTarget.checked)}/>
                    </label>
                    <span>Playback position: <PlaybackPosition gui={this} /></span>
                    <button onClick={() => location.hash = "#" + this.serialize()}>Serialize → URL</button>
                    <ProgressIndicator />
                    {Object.keys(badExamples).length > 0 && <span>Examples: {Object.keys(badExamples).map(txt => <a key={txt} href={badExamples[txt]}
                        onClick={e => {this.deserialize(badExamples[txt].substr(1))}}>{txt}</a>)}</span>}
                </div>
                <div ref={this.setUisDiv}>
                    <div style={{display: "flex", visibility: "hidden"}}>
                        <div style={styles.leftBarCSS} />
                        <div style={{flexGrow: 1}} ref={this.setWidthCalcDiv} />
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

Object.assign(window, {gui:_gui, util, globalConfig, mobx, Data});