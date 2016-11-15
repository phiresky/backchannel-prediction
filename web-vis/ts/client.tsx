import * as React from 'react';
import * as ReactDOM from 'react-dom';
import { autorun, computed, observable, action, extendObservable, useStrict, isObservableArray, asMap } from 'mobx';
useStrict(true);
import { observer } from 'mobx-react';
import * as Waveform from './Waveform';
import * as util from './util';

export const globalConfig = {
    maxColor: "#3232C8",
    rmsColor: "#6464DC",
    leftBarSize: 100,
    zoomFactor: 1.2,
    visualizerHeight: 150,
};


export interface VisualizerProps<T> {
    gui: GUI;
    config: VisualizerConfig;
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
    range: [number, number] | "normalize"
};
export type NumFeatureSVector = NumFeatureCommon & {
    typ: "FeatureType.SVector",
    data: number[]
};
export type NumFeatureFMatrix = NumFeatureCommon & {
    typ: "FeatureType.FMatrix",
    data: number[][]
};
export type Utterances = {
    name: string,
    typ: "utterances",
    data: {from: number | string, to: number | string, text: string, id: string}[]
}
export type NumFeature = NumFeatureSVector | NumFeatureFMatrix;
export type Feature = NumFeature | Utterances;


export type VisualizerConfig  = {min: number, max: number} | "normalize";

interface UIState {
    visualizer: string;
    visualizerConfig: VisualizerConfig,
    feature: string;
}
interface Zoom {
    left: number; right: number;
}

@observer
class InfoVisualizer extends React.Component<{uiState: UIState, zoom: Zoom, gui: GUI}, {}> {
    @observable range = "[-1:1]";
    static ranges: {[name: string]: (f: NumFeature) => VisualizerConfig} = {
        "[-1:1]": (f) => ({min: -1, max: 1}),
        "[0:1]": (f) => ({min: 0, max: 1}),
        "normalize": (f) => "normalize"
    }
    /*@action
    changeRange(evt: React.FormEvent<HTMLSelectElement>) {
        this.range = evt.currentTarget.value;
        console.log(this.range);
        this.props.uiState.visualizerConfig = InfoVisualizer.ranges[this.range](features.get(this.props.uiState.feature)!);        
    }*/
    render() {
        const {uiState, zoom, gui} = this.props;
        const Visualizer = getVisualizer(uiState);
        if (!Visualizer) throw Error("Could not find visualizer " + uiState.visualizer);
        return (
            <div style={{display: "flex"}}>
                <div style={{flexBasis: "content", flexGrow:0, flexShrink: 0, width:globalConfig.leftBarSize+"px"}}>
                    {uiState.feature}
                    {/*<select value={this.range} onChange={this.changeRange.bind(this)} >
                        {Object.keys(InfoVisualizer.ranges).map(k => <option key={k} value={k}>{k}</option>)}
                    </select>*/}
                </div>
                <div style={{flexGrow: 1}}>
                    <Visualizer config={uiState.visualizerConfig} zoom={zoom} feature={features.get(uiState.feature)!} gui={gui} />
                </div>
            </div>
        );

    }
}
@observer
class TextVisualizer extends React.Component<VisualizerProps<Utterances>, {}> {
    // @computed currentlyVisibleTh
    render() {
        const width = this.props.gui.width - 5;
        const pos = this.props.gui.playbackPosition;
        return <div style={{position: "relative", height: "40px"/*, overflowX: "hidden"*/}}>
        {this.props.feature.data.map(utt => {
            const from = +utt.from / state.totalTimeSeconds, to = +utt.to / state.totalTimeSeconds;
            let left = util.getPixelFromPosition(from, 0, width, this.props.zoom);
            let right = util.getPixelFromPosition(to, 0, width, this.props.zoom);
            if ( right < 0 || left > this.props.gui.width) return null;
            const style = {}
            let className = "utterance tooltip";
            if(left < 0) {
                left = 0;
                Object.assign(style, {borderLeft: "none"});
                className += " leftcutoff"; 
            }
            if(right > width) {
                console.log(right, width, utt.to);
                right = width;
                Object.assign(style, {borderRight: "none"});
                className += " rightcutoff";
            }
            if (pos > from && pos <= to)
                className += " visible";
            Object.assign(style, {left:left+"px", width: (right-left)+"px"});
            return <div className={className} key={utt.id} style={style}>
                <span>{utt.text}</span>
                <span className="content"><b/>{utt.text}</span>
            </div>;
        })}
        </div>
    }
}
@observer
class AudioPlayer extends React.Component<{features: NumFeatureSVector[], zoom: Zoom, gui: GUI}, {}> {
    playerBar: HTMLDivElement;
    disposers: (() => void)[] = [];
    audio: AudioContext;
    audioBuffers = new WeakMap<NumFeatureSVector, AudioBuffer>();
    audioSources = [] as AudioBufferSourceNode[];
    duration = 0;
    playing: boolean;
    startedAt: number;
    constructor(props: any) {
        super(props);
        this.audio = new AudioContext();
    }

    updateBar = action(() => {
        if(this.audioSources.length === 0) return;
        this.props.gui.playbackPosition = (this.audio.currentTime - this.startedAt) / this.duration;
        this.playerBar.style.left = util.getPixelFromPosition(this.props.gui.playbackPosition, this.props.gui.left, this.props.gui.width, this.props.zoom) + "px";
        if(this.playing) requestAnimationFrame(this.updateBar);
    });
    stopPlayback() {
        while(this.audioSources.length > 0) {
            this.audioSources.pop()!.stop();
        }
    }

    componentDidMount() {
        const {gui, zoom} = this.props;
        this.disposers.push(autorun(() => this.playerBar.style.left = util.getPixelFromPosition(gui.playbackPosition, gui.left, gui.width, zoom) + "px"));
        window.addEventListener("click", action((event: MouseEvent) => {
            this.stopPlayback();
            this.props.gui.playbackPosition = util.getPositionFromPixel(event.clientX, gui.left, gui.width, zoom)!;
            this.playerBar.style.left = event.clientX + "px";
        }));
        window.addEventListener("keydown", event => {
            if(event.keyCode == 32) {
                event.preventDefault();
            }
        });
        window.addEventListener("keyup", event => {
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
                        audioSource.addEventListener("ended", () => this.playing = false);
                        this.audioSources.push(audioSource);
                    }
                    this.playing = true;
                    requestAnimationFrame(this.updateBar);
                }
            }
        });
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
            <div ref={p => this.playerBar = p} style={{position: "fixed", width: "2px", height: "100vh", top:0, left:-10, backgroundColor:"gray"}} />
        );
    }
}
const features = new Map<string, Feature>();
function getVisualizer(uiState: UIState): Visualizer<any> {
    const feat = features.get(uiState.feature)!;
    if(feat.typ === "FeatureType.SVector") {
        return Waveform.AudioWaveform;
    } else if (feat.typ === "FeatureType.FMatrix") {
        return Waveform.MultiWaveform;
    } else if (feat.typ === "utterances") {
        return TextVisualizer;
    } else throw Error("Can't visualize " + (feat as any).typ);
}
const state = observable({
    uis: [] as UIState[],
    zoom: {
        left: 0, right: 1
    },
    totalTimeSeconds: NaN
});

const socket = new WebSocket(`ws://${location.host.split(":")[0]}:8765`);

socket.onopen = event => { };

function splitIn(count: number, data: Utterances) {
    const feats: Utterances[] = [];
    for(let i = 0; i < count; i++) {
        feats[i] = {name: data.name+"."+i, typ: data.typ, data: []};
    }
    data.data.forEach((utt, i) => feats[i%count].data.push(utt));
    return feats;
}
socket.onmessage = action((event: MessageEvent) => {
    const data: Feature = JSON.parse(event.data);
    console.log(data);
    features.set(data.name, data);
    if(data.typ === "utterances") {
        /*for(const feat of splitIn(5, data)) {
            features.set(feat.name, feat);
            state.uis.push({ feature: feat.name, visualizer: "TextVisualizer", visualizerConfig: "normalize"});
        }*/
        state.uis.push({ feature: data.name, visualizer: "TextVisualizer", visualizerConfig: "normalize"});
    } else {
        let visualizerConfig: VisualizerConfig;
        if(data.range instanceof Array) {
            visualizerConfig = {min: data.range[0], max: data.range[1]};
        } else visualizerConfig = data.range;
        let totalTime;
        if(data.typ === "FeatureType.SVector") {
            totalTime = data.data.length / (data.samplingRate * 1000);
        } else if(data.typ === "FeatureType.FMatrix") {
            totalTime = data.data.length * data.shift / 1000;
        }
        if (totalTime) {
            console.log(totalTime);
            if(isNaN(state.totalTimeSeconds)) state.totalTimeSeconds = totalTime;
            else if(Math.abs((state.totalTimeSeconds - totalTime) / totalTime) > 0.001) {
                console.error("Mismatching times, was ", state.totalTimeSeconds, "but", data.name, "has length", totalTime);
            }
        }
        state.uis.push({ feature: data.name, visualizer: "Waveform", visualizerConfig });
    }
});
@observer
class GUI extends React.Component<{}, {}> {
    @observable widthCalcDiv: HTMLDivElement;
    @observable windowWidth = window.innerWidth;
    @observable playbackPosition = 0;
    @computed
    get left() {
        this.windowWidth;
        return this.widthCalcDiv.getBoundingClientRect().left;
    }
    @computed
    get width() {
        this.windowWidth;
        return this.widthCalcDiv ? this.widthCalcDiv.clientWidth : 100;
    }
    @action
    onWheel(event: MouseWheelEvent) {
            if (!event.ctrlKey) return;
            event.preventDefault();
            const position = util.getPositionFromPixel(event.clientX, this.left, this.width, state.zoom)!;
            const scale = 1/(state.zoom.right - state.zoom.left);
            const scaleChange = event.deltaY > 0 ? globalConfig.zoomFactor : 1/globalConfig.zoomFactor;
            const newScale = scale * scaleChange;
            state.zoom.left -= position;
            state.zoom.right -= position;
            state.zoom.left *= scaleChange; 
            state.zoom.right *= scaleChange;
            state.zoom.left += position;
            state.zoom.right += position;
            state.zoom.right = Math.min(state.zoom.right, 1);
            state.zoom.left = Math.max(state.zoom.left, 0);
    }
    constructor() {
        super();
        window.addEventListener("wheel", e => this.onWheel(e));
        window.addEventListener("resize", action((e: UIEvent) => this.windowWidth = window.innerWidth));
    }
    render() {
        const visibleFeatures = new Set(state.uis.map(ui => ui.feature));
        const visibleAudioFeatures = [...visibleFeatures]
            .map(f => features.get(f)!)
            .filter(f => f.typ === "FeatureType.SVector") as NumFeatureSVector[];
        let audioPlayer
        if(visibleAudioFeatures.length > 0)
            audioPlayer = <AudioPlayer features={visibleAudioFeatures} zoom={state.zoom} gui={this} />;
        return (
            <div>
                <div style={{display: "flex"}}>
                    <div style={{flex: "0 0 content", width:globalConfig.leftBarSize+"px"}}></div>
                    <div style={{flexGrow: 1}} ref={action((e: any) => this.widthCalcDiv = e)} ></div>
                </div>
                {audioPlayer}
                {state.uis.map((p, i) => <InfoVisualizer key={p.feature} uiState={p} zoom={state.zoom} gui={this} />)}
            </div>
        );
    }
}


const gui = ReactDOM.render(<GUI />, document.getElementById("root"));
Object.assign(window, {gui, state, features, util, action});
/*window.addEventListener("wheel", event => {
    if (event.deltaY > 0) canvas.width *= 1.1;
    else canvas.width *= 0.9;
    renderWaveform(canvas, data);
});*/

