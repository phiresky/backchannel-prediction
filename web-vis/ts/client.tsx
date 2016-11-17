import * as React from 'react';
import * as ReactDOM from 'react-dom';
import { autorun, computed, observable, action, extendObservable, useStrict, isObservableArray, asMap } from 'mobx';
useStrict(true);
import { observer } from 'mobx-react';
import * as Waveform from './Waveform';
import * as util from './util';
import DevTools from 'mobx-react-devtools';
import {SocketManager} from './socket';

export const globalConfig = observable({
    maxColor: "#3232C8",
    rmsColor: "#6464DC",
    leftBarSize: 150,
    zoomFactor: 1.2,
    visualizerHeight: 100,
    
});
export class styles {
    @computed static get leftBarCSS() {
        return {flexBasis: "content", flexGrow:0, flexShrink: 0, width:globalConfig.leftBarSize+"px", border:"1px solid", marginRight:"5px"}
    }
}

export interface VisualizerProps<T> {
    gui: GUI;
    uiState: UIState;
    feature: T;
    zoom: Zoom;
}
export interface Visualizer<T> {
    new (props?: VisualizerProps<T>, context?: any): React.Component<VisualizerProps<T>, {}>;
}
export type NumFeatureCommon = {
    name: string,
    samplingRate: number, // in kHz
    shift: number,
    range: [number, number] | null
};
export type NumFeatureSVector = NumFeatureCommon & {
    typ: "FeatureType.SVector",
    data: number[]
};
export type NumFeatureFMatrix = NumFeatureCommon & {
    typ: "FeatureType.FMatrix",
    data: number[][]
};
export type Color = [number, number, number];

export type Utterances = {
    name: string,
    typ: "utterances",
    data: {from: number | string, to: number | string, text: string, id: string, color: Color|null}[]
}
export type Highlights = {
    name: string,
    typ: "highlights",
    data: Highlight[];
}
export type NumFeature = NumFeatureSVector | NumFeatureFMatrix;
export type Feature = NumFeature | Utterances | Highlights;
export type Highlight = {from: number, to: number, color: Color};

export type VisualizerConfig  = "normalizeGlobal" | "normalizeLocal" | "givenRange";

interface UIState {
    uuid: number,
    visualizer: string;
    visualizerConfig: VisualizerConfig,
    feature: string[];
    currentRange: {min: number, max: number} | null
}
let uuid = 0;
interface Zoom {
    left: number; right: number;
}

@observer
class LeftBar extends React.Component<{uiState: UIState, gui: GUI}, {}> {
    static rangeOptions = ["normalizeGlobal", "normalizeLocal", "givenRange"]
    @action changeVisualizerConfig(e: React.SyntheticEvent<HTMLSelectElement>) {
        this.props.uiState.visualizerConfig = e.currentTarget.value as any;
    }
    @action changeFeature(e: React.SyntheticEvent<HTMLSelectElement>, i: number) {
        this.props.uiState.feature[i] = e.currentTarget.value;
    }
    @action remove(i: number) {
        this.props.uiState.feature.splice(i, 1);
    }
    @action add() {
        this.props.uiState.feature.push(this.props.gui.socketManager.features.keys().next().value);
    }
    @action delete() {
        const uis = this.props.gui.uis;
        uis.splice(uis.findIndex(ui => ui.uuid === this.props.uiState.uuid), 1);
    }
    render() {
        let minmax;
        const {uiState, gui} = this.props;
        if(uiState.currentRange) {
            minmax = [
                <pre key="max" style={{position: "absolute", top:0, right:0}}>{util.round1(uiState.currentRange.max)}</pre>,
                <pre key="min" style={{position: "absolute", bottom:0, right:0}}>{util.round1(uiState.currentRange.min)}</pre>,
                <div key="sel" style={{position: "absolute", bottom:0, top:0, right:0, display:"flex", justifyContent:"center", flexDirection:"column"}}>
                    <select value={uiState.visualizerConfig} onChange={this.changeVisualizerConfig.bind(this)}>
                        {LeftBar.rangeOptions.map(op => <option key={op} value={op}>{op}</option>)}
                    </select>
                </div>
            ];
        } else minmax = "";
        return (
            <div style={{position: "relative", width:"100%", height:"100%"}}>
                <div style={{position: "absolute", top:0, left:0, zIndex: 1}}>
                    {uiState.feature.map((feature, i) => 
                        <div key={i} ><button onClick={e => this.remove(i)}>-</button>
                            <select value={feature} onChange={e => this.changeFeature(e, i)}>
                                {[...gui.socketManager.features.keys()].map(f => <option key={f} value={f}>{f}</option>)}
                            </select>
                        </div>
                    )}
                    <button onClick={e => this.add()}>+</button>
                </div>
                {minmax}
                <div style={{position: "absolute", bottom:0, left:0}}><button onClick={this.delete.bind(this)}>×</button></div>
            </div>
        );
    }
}
@observer
class InfoVisualizer extends React.Component<{uiState: UIState, zoom: Zoom, gui: GUI}, {}> {
    render() {
        const {uiState, zoom, gui} = this.props;
        
        return (
            <div style={{display: "flex", marginBottom:"10px"}}>
                <div style={styles.leftBarCSS}>
                    <LeftBar gui={gui} uiState={uiState} />
                </div>
                <div style={{flexGrow: 1}}>
                    <Waveform.HighlightOverlayVisualizer uiState={uiState} gui={gui} zoom={zoom} feature={uiState.feature.map(f => gui.socketManager.features.get(f)!)}/>
                </div>
            </div>
        );

    }
}
@observer
class TextVisualizer extends React.Component<VisualizerProps<Utterances>, {}> {
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
            let left = util.getPixelFromPosition(from, 0, width, this.props.zoom);
            let right = util.getPixelFromPosition(to, 0, width, this.props.zoom);
            if ( right < 0 || left > this.props.gui.width) return null;
            const style = {height: "20px"};
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
            Object.assign(style, {left:left+"px", width: (right-left - padding*2)+"px", padding: padding +"px"});
            return <div className={className} key={utt.id} style={style}
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
        let left = util.getPixelFromPosition(from, 0, width, this.props.zoom);
        let right = util.getPixelFromPosition(to, 0, width, this.props.zoom);
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
            {this.playbackTooltip !== null && this.props.gui.audioPlayer.playing && <div style={{position: "relative", height: "0px", width:"100%"}}>{this.getTooltip(this.playbackTooltip)}</div>}
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
@observer
class AudioPlayer extends React.Component<{features: NumFeatureSVector[], zoom: Zoom, gui: GUI}, {}> {
    playerBar: HTMLDivElement; setPlayerBar = (p: HTMLDivElement) => this.playerBar = p;
    disposers: (() => void)[] = [];
    audio: AudioContext;
    audioBuffers = new WeakMap<NumFeatureSVector, AudioBuffer>();
    audioSources = [] as AudioBufferSourceNode[];
    duration = 0;
    @observable
    playing: boolean;
    startedAt: number;
    constructor(props: any) {
        super(props);
        this.audio = new AudioContext();
    }

    center() {
        const zoom = this.props.gui.zoom;
        const w = zoom.right - zoom.left;
        let pos = this.props.gui.playbackPosition;
        if (pos - w/2 < 0) pos = w/2;
        if (pos + w/2 > 1) pos = 1 - w/2;
        zoom.left = pos - w/2; zoom.right = pos + w/2;
    }

    updatePlaybackPosition = action("updatePlaybackPosition", () => {
        if(this.audioSources.length === 0) return;
        this.props.gui.playbackPosition = (this.audio.currentTime - this.startedAt) / this.duration;
        if(this.props.gui.followPlayback) this.center();
        if(this.playing) requestAnimationFrame(this.updatePlaybackPosition);
    });
    updatePlayerBar = () => {
        const x = util.getPixelFromPosition(this.props.gui.playbackPosition, this.props.gui.left, this.props.gui.width, this.props.zoom);
        this.playerBar.style.transform = "translateX("+x+"px)";
    }
    stopPlayback() {
        while(this.audioSources.length > 0) {
            this.audioSources.pop()!.stop();
        }
    }

    onKeyUp = action("onKeyUp", (event: KeyboardEvent) => {
        if(event.keyCode == 32) {
            event.preventDefault();
            if(this.audioSources.length > 0) {
                this.stopPlayback();
            } else {
                for(const feature of this.props.features) {
                    const buffer = this.audioBuffers.get(feature)!;
                    const audioSource = this.audio.createBufferSource();
                    audioSource.buffer = buffer;
                    audioSource.connect(this.audio.destination);
                    audioSource.start(0, this.props.gui.playbackPosition * buffer.duration);
                    this.startedAt = this.audio.currentTime - this.props.gui.playbackPosition * buffer.duration;
                    audioSource.addEventListener("ended", action("audioEnded", () => this.playing = false));
                    this.audioSources.push(audioSource);
                }
                this.playing = true;
                requestAnimationFrame(this.updatePlaybackPosition);
            }
        }
    })
    onKeyDown = (event: KeyboardEvent) => {
        if(event.keyCode == 32) {
            event.preventDefault();
        }
    }
    onClick = action("clickSetPlaybackPosition", (event: MouseEvent) => {
        if(event.clientX < this.props.gui.left) return;
        event.preventDefault();
        const x = util.getPositionFromPixel(event.clientX, this.props.gui.left, this.props.gui.width, this.props.zoom)!;
        this.stopPlayback();
        this.props.gui.playbackPosition = Math.max(x, 0);
    });

    componentDidMount() {
        const {gui, zoom} = this.props;
        this.disposers.push(autorun("updatePlayerBar", this.updatePlayerBar));
        const uisDiv = this.props.gui.uisDiv;
        uisDiv.addEventListener("click", this.onClick);
        window.addEventListener("keydown", this.onKeyDown);
        window.addEventListener("keyup", this.onKeyUp);
        this.disposers.push(() => uisDiv.removeEventListener("click", this.onClick));
        this.disposers.push(() => window.removeEventListener("keydown", this.onKeyDown));
        this.disposers.push(() => window.removeEventListener("keyup", this.onKeyUp));
        this.disposers.push(() => this.stopPlayback());
    }
    componentWillUnmount() {
        console.log("pp", "willunmonut");
        for(const disposer of this.disposers) disposer();
    }
    render() {
        for(const feature of this.props.features) {
            if(this.audioBuffers.has(feature)) continue;
            console.log("creating buffer for "+feature.name);
            const audioBuffer = this.audio.createBuffer(1, feature.data.length, feature.samplingRate * 1000);
            const arr = Float32Array.from(feature.data, v => v / 2 ** 15);
            audioBuffer.copyToChannel(arr, 0);
            this.duration = audioBuffer.duration;
            this.audioBuffers.set(feature, audioBuffer);
        }
        return (
            <div ref={this.setPlayerBar} style={{position: "fixed", width: "2px", height: "100vh", top:0, left:0, backgroundColor:"gray"}} />
        );
    }
}

export function GetVisualizer(props: VisualizerProps<Feature>): JSX.Element {
    if(props.feature.typ === "FeatureType.SVector" || props.feature.typ === "FeatureType.FMatrix") {
        return <Waveform.AudioWaveform feature={props.feature} uiState={props.uiState} gui={props.gui} zoom={props.zoom} />;
    } else if (props.feature.typ === "utterances") {
        return <TextVisualizer feature={props.feature} uiState={props.uiState} gui={props.gui} zoom={props.zoom} />;
    } else if(props.feature.typ === "highlights") {
        return <Waveform.HighlightsVisualizer feature={props.feature} uiState={props.uiState} gui={props.gui} zoom={props.zoom} />;
    } else throw Error("Can't visualize " + (props.feature as any).typ);
}


function splitIn(count: number, data: Utterances) {
    const feats: Utterances[] = [];
    for(let i = 0; i < count; i++) {
        feats[i] = {name: data.name+"."+i, typ: data.typ, data: []};
    }
    data.data.forEach((utt, i) => feats[i%count].data.push(utt));
    return feats;
}

@observer
class ConversationSelector extends React.Component<{gui: GUI}, {}> {
    setConversation = action("setConversation", (e: React.SyntheticEvent<HTMLInputElement>) =>
        this.props.gui.conversation = e.currentTarget.value);
    render() {
        return (<div style={{display:"inline-block"}}>
            <input list="conversations" value={this.props.gui.conversation} onChange={this.setConversation} />
            <datalist id="conversations">
                {this.props.gui.socketManager.conversations.map(c => <option key={c} value={c}/>)}
            </datalist>
            <button onClick={c => gui.loadConversation()}>Load</button>
        </div>)
    }
}
@observer
export class GUI extends React.Component<{}, {}> {
    @observable widthCalcDiv: HTMLDivElement;
    @observable windowWidth = window.innerWidth;
    @observable playbackPosition = 0;
    @observable followPlayback = true;

    @observable conversation = "sw2807";
    @observable uis = [] as UIState[];
    @observable zoom = {
        left: 0, right: 1
    };
    @observable totalTimeSeconds = NaN;

    audioPlayer: AudioPlayer; setAudioPlayer = action("setAudioPlayer", (a: AudioPlayer) => this.audioPlayer = a);
    uisDiv: HTMLDivElement; setUisDiv = action("setUisDiv", (e: HTMLDivElement) => this.uisDiv = e);
    socketManager: SocketManager;

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
    loadConversation() {
        this.uis = [];
        this.zoom.left = 0; this.zoom.right = 1;
        this.totalTimeSeconds = NaN;
        this.socketManager.loadConversation(this.conversation);
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
    onFeatureReceived(feature: Feature) {
        if(feature.typ === "utterances") {
            this.uis.push({ uuid: uuid++, feature: [feature.name], visualizer: "TextVisualizer", visualizerConfig: "normalizeLocal", currentRange: null});
        } else if (feature.typ === "highlights") {
            
        } else {
            let visualizerConfig: VisualizerConfig;
            if(feature.range instanceof Array) {
                visualizerConfig = "givenRange";
            } else visualizerConfig = "normalizeLocal";
            let totalTime;
            if(feature.typ === "FeatureType.SVector") {
                totalTime = feature.data.length / (feature.samplingRate * 1000);
            } else if(feature.typ === "FeatureType.FMatrix") {
                totalTime = feature.data.length * feature.shift / 1000;
            }
            if (totalTime) {
                if(isNaN(this.totalTimeSeconds)) this.totalTimeSeconds = totalTime;
                else if(Math.abs((this.totalTimeSeconds - totalTime) / totalTime) > 0.001) {
                    console.error("Mismatching times, was ", this.totalTimeSeconds, "but", feature.name, "has length", totalTime);
                }
            }
            this.uis.push({ uuid: uuid++, feature: [feature.name], visualizer: "Waveform", visualizerConfig, currentRange: null});
        }
    }
    constructor() {
        super();
        this.socketManager = new SocketManager(this, `ws://${location.host.split(":")[0]}:8765`);
        window.addEventListener("wheel", e => this.onWheel(e));
        window.addEventListener("resize", action("windowResize", (e: UIEvent) => this.windowWidth = window.innerWidth));
    }
    render(): JSX.Element {
        const visibleFeatures = new Set(this.uis.map(ui => ui.feature).reduce((a,b) => (a.push(...b),a), []));
        const visibleAudioFeatures = [...visibleFeatures]
            .map(f => this.socketManager.features.get(f)!)
            .filter(f => f.typ === "FeatureType.SVector") as NumFeatureSVector[];
        let audioPlayer
        if(visibleAudioFeatures.length > 0)
            audioPlayer = <AudioPlayer features={visibleAudioFeatures} zoom={this.zoom} gui={this} ref={this.setAudioPlayer} />;
        return (
            <div>
                <div style={{margin: "10px"}}>
                    <ConversationSelector gui={this} />
                    <label style={{marginLeft: "10px"}}>Follow playback:
                        <input type="checkbox" checked={this.followPlayback}
                            onChange={action("changeFollowPlayback", (e: React.SyntheticEvent<HTMLInputElement>) => this.followPlayback = e.currentTarget.checked)}/>
                    </label>
                </div>
                <div ref={this.setUisDiv}>
                    <div style={{display: "flex", visibility: "hidden"}}>
                        <div style={styles.leftBarCSS} />
                        <div style={{flexGrow: 1}} ref={action("setWidthCalcDiv", (e: any) => this.widthCalcDiv = e)} />
                    </div>
                    
                    {this.uis.map((p, i) => <InfoVisualizer key={p.uuid} uiState={p} zoom={this.zoom} gui={this} />)}
                </div>
                {audioPlayer}
                <DevTools />
            </div>
        );
    }
}


const gui = ReactDOM.render(<GUI />, document.getElementById("root")) as GUI;
Object.assign(window, {gui, util, action, globalConfig});
