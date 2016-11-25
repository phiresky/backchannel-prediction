import * as React from 'react';
import * as ReactDOM from 'react-dom';
import { autorun, computed, observable, action, extendObservable, useStrict, isObservableArray, asMap, toJS, runInAction } from 'mobx';
useStrict(true);
import { observer } from 'mobx-react';
import * as Waveform from './Waveform';
import * as util from './util';
import DevTools from 'mobx-react-devtools';
import * as s from './socket';
import * as LZString from 'lz-string';
import * as highlights from './Highlights';

export const globalConfig = observable({
    maxColor: "#3232C8",
    rmsColor: "#6464DC",
    leftBarSize: 200,
    zoomFactor: 1.2,
    visualizerHeight: 100,
    defaultConversation: "sw2807"
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
    data: ArrayLike<number>,
    dtype: "int16"
};

export type NumFeatureFMatrix = NumFeatureCommon & {
    typ: "FeatureType.FMatrix",
    data: number[][],
    dtype: "float32"
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
export type Highlight = {from: number, to: number, color: Color, text?: string};

export type VisualizerConfig  = "normalizeGlobal" | "normalizeLocal" | "givenRange";

export interface SingleUIState {
    visualizer: VisualizerChoice;
    feature: string;
    config: VisualizerConfig;
    currentRange: {min: number, max: number} | null;
}
interface UIState {
    uuid: number,
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
        const state = this.props.gui.getDefaultUIState(await this.props.gui.getFeature(e.currentTarget.value).toPromise())
        runInAction("changeFeature"+i, () => this.props.uiState.features[i] = state);
    }
    @action remove(i: number) {
        this.props.uiState.features.splice(i, 1);
        if(this.props.uiState.features.length === 0) {
            const uis = this.props.gui.uis;
            uis.splice(uis.findIndex(ui => ui.uuid === this.props.uiState.uuid), 1);
        }
    }
    @action async add() {
        const gui = this.props.gui;
        const state = gui.getDefaultUIState(await gui.getFeature((await gui.getFeatures().toPromise()).defaults[0]).toPromise());
        runInAction(() => this.props.uiState.features.push(state));
    }

    render() {
        let minmax;
        const {uiState, gui} = this.props;
        const firstWithRange = uiState.features.find(props => props.currentRange !== null);
        if(firstWithRange && firstWithRange.currentRange) {
            minmax = [
                <pre key="max" style={{position: "absolute", margin:0, top:0, right:0}}>{util.round1(firstWithRange.currentRange.max)}</pre>,
                <pre key="min" style={{position: "absolute", margin:0, bottom:0, right:0}}>{util.round1(firstWithRange.currentRange.min)}</pre>,
            ];
        } else minmax = "";
        const VisualizerChoices = (props:{info: SingleUIState}) => {
            const feature = gui.getFeature(props.info.feature).data;
            if(!feature) return <span/>;
            const c = getVisualizerChoices(feature);
            if(c.length > 1) return <select value={props.info.visualizer} onChange={e => this.changeVisualizer(props.info, e.currentTarget.value)}>
                {c.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
            else return <span/>;
        }
        const features = gui.getFeatures().data;
        return (
            <div className="left-bar" style={{position: "relative", width:"100%", height:"100%"}}>
                <div style={{position: "absolute", top:0, bottom:0, left:0, zIndex: 1, display:"flex", justifyContent:"center", flexDirection:"column",alignItems:"flex-start"}}>
                    {uiState.features.map((info, i) => 
                        <div key={i}><button onClick={e => this.remove(i)}>-</button>
                            {features && <select style={{maxWidth:"7.5em"}} value={info.feature} onChange={e => this.changeFeature(e, i)}>
                                        {Object.keys(features.categories).map(category => 
                                            <optgroup key={category} label={category}>{features.categories[category].map(f => <option key={f.toString()} value={f.toString()}>{f}</option>)}</optgroup>
                                        )}
                                    </select>
                            }
                            {info.currentRange&&
                                <select style={{maxWidth:"7.5em"}} value={info.config} onChange={e => this.changeVisualizerConfig(info, e.currentTarget.value)}>
                                    {LeftBar.rangeOptions.map(op => <option key={op} value={op}>{op}</option>)}
                                </select>
                            }
                            <VisualizerChoices info={info} />
                        </div>
                        
                    )}
                    <button onClick={e => this.add()}>+</button>
                </div>
                {minmax}
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
                    <highlights.OverlayVisualizer gui={gui} uiStates={uiState.features} />
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
            let left = util.getPixelFromPosition(from, 0, width, this.props.gui.zoom);
            let right = util.getPixelFromPosition(to, 0, width, this.props.gui.zoom);
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
            {this.playbackTooltip !== null && this.props.gui.audioPlayer && this.props.gui.audioPlayer.playing && <div style={{position: "relative", height: "0px", width:"100%"}}>{this.getTooltip(this.playbackTooltip)}</div>}
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
        this.disposers.push(() => (this.audio as any).close());
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
    onClick = (event: MouseEvent) => {
        if(event.clientX < this.props.gui.left) return;
        event.preventDefault();
        const x = util.getPositionFromPixel(event.clientX, this.props.gui.left, this.props.gui.width, this.props.zoom)!;
        this.stopPlayback();
        runInAction("clickSetPlaybackPosition", () => this.props.gui.playbackPosition = Math.max(x, 0));
    }

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
type VisualizerChoice = "Waveform"|"Darkness"|"Text"|"Highlights";

export function getVisualizerChoices(feature: Feature): VisualizerChoice[] {
    if(!feature) return [];
    if(feature.typ === "FeatureType.SVector" || feature.typ === "FeatureType.FMatrix") {
        return ["Waveform", "Darkness"];
    } else if (feature.typ === "utterances") {
        return ["Text"];
    } else if(feature.typ === "highlights") {
        return ["Highlights"];
    } else throw Error("Can't visualize " + (feature as any).typ);
}

@observer
export class ChosenVisualizer extends React.Component<VisualizerProps<Feature>,{}> {
    static visualizers = {
        "Waveform": Waveform.AudioWaveform,
        "Darkness": Waveform.Darkness,
        "Text": TextVisualizer,
        "Highlights": highlights.HighlightsVisualizer
    }
    render() {
        const Visualizer = ChosenVisualizer.visualizers[this.props.uiState.visualizer] as Visualizer<Feature>;
        return <Visualizer {...this.props} />;
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
    conversation = "sw2001";
    render() {
        const convos = this.props.gui.getConversations().data;
        return (<div style={{display:"inline-block"}}>
            <input list="conversations" value={this.conversation} onChange={e => this.conversation = e.currentTarget.value} />
            {convos && <datalist id="conversations">{convos.map(c => <option key={c as any} value={c as any}/>)}</datalist>}
            <button onClick={c => this.props.gui.loadConversation(this.conversation)}>Load</button>
            <button onClick={c => this.props.gui.loadRandomConversation()}>RND</button>
        </div>)
    }
}

const badExamples: {[name: string]: string} = {
    "too early": "#N4IgDgNghgngRlAxgawAoHsDOBLALt9AOxAC4AGAOgHYqA2AJgE4yyAOAZkcYBZb2BWWgBoQAM3QQI6AO6po8JMlKioETAFMRiIgDd1AJ0xR8RUiEzT6rMlRAiArtkykA2qHuOAJqQCMwserG9vrqriBQnohQduGRUBRwiCAAuiI6Tvaq2ABeBmYA6lB64voAtjHpmJkQOQYAwkSi2ADmZs3YeoQASlCEzaFawSGEuD19oSSgpdjEJAC07PR0rCKlUAAepIvLAL47Qu5evlQiooG4wRMu5pbWVHMAgilpGVm5+mYAKurruABqrxq7wqgNq+gahCarRIIEI6DKb3UABl0FEIDFEEN1CMxv1SIR7JJ9odsN4SD4VgEgiEwmA8IgABbREQRKIJJKpECVapggpFdQlcovKqI8GNFpmOEIoHI1GqDFYnG9PGTEDTWb0fxrTYkRhUPYHEAeUm+Rinc6XWkyAzM2JsxLPLmg4EwwrFeFCp0imViyESmFStYylFohX6YajZUTKYzUhzShkLiMWgUxOsHycOhMVYbUiUdjsVjceiCdgMRg+fhkTgGklk+hkc3Uq52uAxVlwdmO7mivnusog71giFQtodbG4gYgTHh7GR8akGOzBZLWiU7VbVesWtGo4keg+JsXGkka4WKw2OYAIW7zryMO+vwBQ+Bwp57xH-thHsRIflg1nJUFxIAkiUNY163oI9LVPcB6QZNsWUiTsHU5HsfT7AUPUHd96nFaFv2lWo-3RACI0nRc1VjEgBBzHUOB3CDSHodhoJPa4wGtfRENbLs0LvD5XX5QUcNFT8CMDX85VI6dFXnFUlzjBMUyobhWH4HxGH4TgOHYGw6LzChuG4LhuHYbhtJsAQyD8PZOWydB0HKVUIAFXBDJoXgqDIfh6As2huCoEtuBEfQWgZdzyGoVhmBMwKbIrHwWNofYQFwdBcFUT5sFKdQAGV1G0QhPGcEhaBYCgM20qh+B2IA",
    "very delayed": "#N4IgDgNghgngRlAxgawAoHsDOBLALt9AOxAC4AGAOgBYqyBWAdgDYAmMgZgA4BOKpsluwA0IAGboIEdAHdU0eEmSlRUCJgCmIxEQBu6gE6Yo+IqRCZpLTmQYgRAV2yZSAbVD3HAE1IBGJiNF1Y3t9dVcQKE9EKDsIqKgKOEQQAF0RHSd7VWwALwMzAHUoPXF9AFtYjMwsiFyDAGEiUWwAczNCdHLsvIAZdGiIWMQQ0MJcACUoQhawklAy7GISAFomHxYqETKoAA9SJm5BAF8joXcvXwYAoNwQ2ZdzS2sGZYBBVPTM7vySEAAVdQ7XAANS+tTy+kqYLq+kahGabV+HS64PUfQGQxG6jGk2ms0I9kkp3O2G8JB8nGuwVC4TAeEQAAsYiJItFEsk0iAqjUYYViupShVPtVvrCmq12p1tqj0apMfpRhMpjNSPNFqR2CwtrtfGQfCcziAPKTfNwqbcaSQHmAZAZmXE2UkPlzoRC+SUpVCRaixfCJUipd9ZYMtFiccrZmqlstKGRuNxOIw49x2HQqCwmAx1tq9uQKCxuExOFQGNw40WeFRCwaSWS2Oa7uFWXBYs32c7uaL3QLPcKeRC4QjJSi6sH5YrcSq5iAFtG6HHhDOdSRmAma0aLiQWD4G5aHhYrDZlgAhDuun7-QEg8+Qvuiwf+kDI6Wj-py0MK7FKvGkAlEw3GnWWpiDcjZWuA9IMi2LJRHA7acp2PrdoKXr9g04qIk+gYym+IYgMMn7hj+06zhqlJLrmnAMOugGkIIu73OAtr6NBDpwU6CE3shvYut6MIPphz5Brh45fpOkYzuqKyxqwLAMAw7BUOw6xUD4dDrIu2y5pQNB0JqFJWOsBw2KmJycjk6DoBU04QAKuCkDpXBkGQVZkEWLlxi5Ij6K0DL2XmVDznQxYCGwfh0EwEWnCAuDoLgqh-NgZTqAAyuo2iEJ4zgkPwlA+Km7AMHQRxAA",
    "double, only when silence before": "#N4IgDgNghgngRlAxgawAoHsDOBLALt9AOxAC4AGAOgBYAOGqgZgDYqqBOAdiYEYmBWAEzcANCABm6CBHQB3VNHhJkpMVAiYApqMREAbhoBOmKPiKkQmGQJpkOIUQFdsmUgG1QDpwBNSvUWI0TBwMNNxAoL0Qoe3DIqAo4RBAAXVFdZwc1bAAvQ3MAdSh9CQMAWxj0zEyIHMMAYSIxbABzc2bsfUIAJShCZtDtYJDCXB6+0JJQUuxiEgBaBgEuGlFSqAAPUkXlgF8d4Q9vXw5-QNxgidcLKxsOOYBBFLSMrNyDcwAVDXXcADUXmpvCoA2oGBqEJqtEggQjoMqvDQAGXQUQgMUQQw0IzG-VIhAcUn2h2wPhI3BW4jOFzCYDwiAAFtFRBEogkkqkQJVqqCCkUNCVys8qgiwY0WuZYfDAUiUWp0Zjsb1cZMQNNZotVhtfGQyHsDiBPCTfGxTkEQjTZIYmbFWYknpyQUDoYVinDBQ7hdLRRDxdDJWtpcjUfKDMNRkqJlMZqQ5pQyGw2DQmNYaNwdbQ0wwGJrNuQKMxuHx2EwOEWWEwGGRuHriaSBGRTedzSQriy4DE22z7VyRbzXWVgZ7QeDIW0OlicQMQBjQ1jw+NSFHZgslkwKWtc9s1zWDUcSEJG9SW9drLY5gAhbuOvLQr4-f5DoFC7lvEe+mFuhFBuWDWeKhckPihL6oadYCIezZXLSuAMu2zKRHAXYcj2Xp9vybqDi+9RilCH5SrU35or+YaTouqrRiQyY5qQbB8DuoGkAI2aUmalzgJaBhwTaiF2sh17vM6fICphIpvrh-pfrKRHTgq87KkuMZxjwa5MLRvAlrYZAMCaqpanmzCsHwXC2AIqnWBw1Y7By2ToOg5QqhA-K4KQlC0MmthUEsRnaRZtGiAYLT0s5ebsDQHBJkWSxJqwFn7CAuDoLgagfNgpQaAAyhoOiEF4LiUTqFDcAwfAMKWOxAA",
}
enum LoadingState {
    NotConnected, Connected, Loading, Complete
}
let MaybeAudioPlayer = observer<{gui:GUI}>(function MaybeAudioPlayer({gui}: {gui: GUI}) {
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

    loadingState = LoadingState.NotConnected;

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
        if(this.audioPlayer) this.audioPlayer.stopPlayback();
        if(this.loadingState === LoadingState.Loading) {
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
            this.conversation = convID;
            this.uis = [];
            this.zoom.left = 0; this.zoom.right = 1;
            this.totalTimeSeconds = NaN;
        });
        const features = await this.socketManager.getFeatures(convID).toPromise();
        for(const featureID of features.defaults) this.onFeatureReceived(await this.getFeature(featureID).toPromise());
    }
    async verifyConversationID(id: string): Promise<s.ConversationID> {
        const convos = await this.socketManager.getConversations().toPromise();
        if(convos.indexOf(id as any) >= 0) return id as any;
        throw Error("unknown conversation " + id);
    }
    async loadRandomConversation() {
        this.loadConversation(util.randomChoice(await this.getConversations().toPromise()) as any);
    }
    @action onFeatureReceiveDone() {
        if(this.stateAfterLoading) Object.assign(this, this.stateAfterLoading);
        this.stateAfterLoading = null;
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
    getDefaultUIState(feature: Feature): SingleUIState {
        let visualizerConfig: VisualizerConfig;
        if(isNumFeature(feature) && feature.range instanceof Array) {
            visualizerConfig = "givenRange";
        } else visualizerConfig = "normalizeLocal";
        return {
            feature: feature.name,
            visualizer: getVisualizerChoices(feature)[0],
            config: visualizerConfig,
            currentRange: null,
        }
    }
    @action
    onFeatureReceived(feature: Feature) {
        if(feature.typ === "utterances") {
            this.uis.push({ uuid: uuid++, features: [{feature: feature.name, visualizer: "Text", config: "normalizeLocal", currentRange: null}]});
        } else if (feature.typ === "highlights") {
            if(feature.name.includes(".")) {
                const addTo = feature.name.split(".")[0];
                this.uis.filter(ui => ui.features[0].feature.slice(-1) === addTo.slice(-1)).forEach(ui => ui.features.push(this.getDefaultUIState(feature)));
            }
        } else {
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
            this.uis.push({ uuid: uuid++, features: [this.getDefaultUIState(feature)]});
        }
    }
    onSocketOpen() {
        this.loadingState = LoadingState.Connected;
        if(location.hash.length > 1) {
            this.deserialize(location.hash.substr(1));
        } else {
            this.loadConversation(globalConfig.defaultConversation);
        }
    }
    constructor() {
        super();
        this.socketManager = new s.SocketManager(`ws://${location.host.split(":")[0]}:8765`);
        window.addEventListener("wheel", e => this.onWheel(e));
        window.addEventListener("resize", action("windowResize", (e: UIEvent) => this.windowWidth = window.innerWidth));
        window.addEventListener("hashchange", action("hashChange", e => this.deserialize(location.hash.substr(1))));
    }
    getFeature(id: string|s.FeatureID) {
        const f = this.socketManager.getFeature(this.conversation, id as any as s.FeatureID);
        if(!f) throw Error("unknown feature "+ name);
        return f;
    }
    getFeatures() {
        return this.socketManager.getFeatures(this.conversation);
    }
    getConversations() {
        return this.socketManager.getConversations();
    }
    render(): JSX.Element {
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
                    Examples: {Object.keys(badExamples).map(txt => <a key={txt} href={badExamples[txt]}
                        onClick={e => {this.deserialize(badExamples[txt].substr(1))}}>{txt}</a>)}
                </div>
                <div ref={this.setUisDiv}>
                    <div style={{display: "flex", visibility: "hidden"}}>
                        <div style={styles.leftBarCSS} />
                        <div style={{flexGrow: 1}} ref={this.setWidthCalcDiv} />
                    </div>
                    
                    {this.uis.map((p, i) => <InfoVisualizer key={p.uuid} uiState={p} zoom={this.zoom} gui={this} />)}
                </div>
                <MaybeAudioPlayer gui={this} />
                <DevTools />
            </div>
        );
    }
}



const gui = ReactDOM.render(<GUI />, document.getElementById("root")) as GUI;
Object.assign(window, {gui, util, action, globalConfig, toJS});