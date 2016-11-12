import * as React from 'react';
import * as ReactDOM from 'react-dom';
import { autorun, observable, action, extendObservable, useStrict, isObservableArray, asMap } from 'mobx';
useStrict(true);
import { observer } from 'mobx-react';
import * as Waveform from './Waveform';
import * as util from './util';

export const globalConfig = {
    maxColor: "#3232C8",
    rmsColor: "#6464DC",
    leftBarSize: 100,
    zoomFactor: 1.2
};


export interface VisualizerProps<T> {
    config: VisualizerConfig;
    feature: T;
    zoom: Zoom;
}
export interface Visualizer<T> {
    new (props?: VisualizerProps<T>, context?: any): React.Component<VisualizerProps<T>, {}>;
}
type Message = NumFeature;
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
export type NumFeature = NumFeatureSVector | NumFeatureFMatrix;


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
class InfoVisualizer extends React.Component<{uiState: UIState, zoom: Zoom}, {}> {
    @observable range = "[-1:1]";
    static ranges: {[name: string]: (f: NumFeature) => VisualizerConfig} = {
        "[-1:1]": (f) => ({min: -1, max: 1}),
        "[0:1]": (f) => ({min: 0, max: 1}),
        "normalize": (f) => "normalize"
    }
    @action
    changeRange(evt: React.FormEvent<HTMLSelectElement>) {
        this.range = evt.currentTarget.value;
        console.log(this.range);
        this.props.uiState.visualizerConfig = InfoVisualizer.ranges[this.range](features.get(this.props.uiState.feature)!);        
    }
    render() {
        const {uiState, zoom} = this.props;
        const Visualizer = getVisualizer(uiState);
        if (!Visualizer) throw Error("Could not find visualizer " + uiState.visualizer);
        return (
            <div style={{display: "flex"}}>
                <div style={{flexBasis: "content", width:globalConfig.leftBarSize+"px"}}>
                    {uiState.feature}
                    {/*<select value={this.range} onChange={this.changeRange.bind(this)} >
                        {Object.keys(InfoVisualizer.ranges).map(k => <option key={k} value={k}>{k}</option>)}
                    </select>*/}
                </div>
                <div style={{flexGrow: 1}}>
                    <Visualizer config={uiState.visualizerConfig} zoom={zoom} feature={features.get(uiState.feature)!} />
                </div>
            </div>
        );

    }
}
@observer
class AudioPlayer extends React.Component<{features: NumFeatureSVector[], zoom: Zoom}, {}> {
    playerBar: HTMLDivElement;
    disposers: (() => void)[] = [];
    audio: AudioContext;
    audioBuffers = new WeakMap<NumFeatureSVector, AudioBuffer>();
    audioSources = [] as AudioBufferSourceNode[];
    position = 0;
    duration = 0;
    playing: boolean;
    startedAt: number;
    div: HTMLDivElement;
    get left() {
        return this.div.getBoundingClientRect().left;
    }
    constructor(props: any) {
        super(props);
        this.audio = new AudioContext();
    }

    updateBar = () => {
        if(this.audioSources.length === 0) return;
        this.position = (this.audio.currentTime - this.startedAt) / this.duration;
        this.playerBar.style.left = util.getPixelFromPosition(this.position, this.left, this.div.clientWidth, this.props.zoom) + "px";
        if(this.playing) requestAnimationFrame(this.updateBar);
    }
    stopPlayback() {
        while(this.audioSources.length > 0) {
            this.audioSources.pop()!.stop();
        }
    }

    componentDidMount() {
        this.disposers.push(autorun(() => this.playerBar.style.left = util.getPixelFromPosition(this.position, this.left, this.div.clientWidth, this.props.zoom) + "px"));
        window.addEventListener("click", event => {
            this.stopPlayback();
            this.position = util.getPositionFromPixel(event.clientX, this.left, this.div.clientWidth, this.props.zoom)!;
            this.playerBar.style.left = event.clientX + "px";
        });
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
                        audioSource.start(0, this.position * buffer.duration);
                        this.startedAt = this.audio.currentTime - this.position * buffer.duration;
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
            <div style={{display: "flex"}}>
                <div style={{flexBasis: "content", width:globalConfig.leftBarSize+"px"}}>
                </div>
                <div style={{flexGrow: 1}} ref={e => this.div = e} >
                    <div ref={p => this.playerBar = p} style={{position: "fixed", width: "2px", height: "100vh", top:0, left:-10, backgroundColor:"gray"}} />
                </div>
            </div>
        );
    }
}
const features = new Map<string, NumFeature>();
function getVisualizer(uiState: UIState): Visualizer<any> {
    const feat = features.get(uiState.feature)!;
    if(feat.typ === "FeatureType.SVector") {
        return Waveform.AudioWaveform;
    } else return Waveform.MultiWaveform;
}
const state = observable({
    uis: [] as UIState[],
    zoom: {
        left: 0, right: 1
    }
});

const socket = new WebSocket(`ws://${location.host.split(":")[0]}:8765`);

socket.onopen = event => { };

socket.onmessage = action((event: MessageEvent) => {
    const data: Message = JSON.parse(event.data);
    console.log(data);
    features.set(data.name, data);
    let visualizerConfig: VisualizerConfig;
    if(data.range instanceof Array) {
        visualizerConfig = {min: data.range[0], max: data.range[1]};
    } else visualizerConfig = data.range;
    state.uis.push({ feature: data.name, visualizer: "Waveform", visualizerConfig });
});
@observer
class GUI extends React.Component<{}, {}> {
    @action
    onWheel(event: MouseWheelEvent) {
            if (!(event.target instanceof HTMLCanvasElement)) return;
            event.preventDefault();
            const position = util.getPositionFromPixel(event.clientX, event.target.getBoundingClientRect().left, event.target.width, state.zoom)!;
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
        window.addEventListener("wheel", this.onWheel);
    }
    render() {
        const visibleFeatures = new Set(state.uis.map(ui => ui.feature));
        const visibleAudioFeatures = [...visibleFeatures]
            .map(f => features.get(f)!)
            .filter(f => f.typ === "FeatureType.SVector") as NumFeatureSVector[];
        let audioPlayer
        if(visibleAudioFeatures.length > 0)
            audioPlayer = <AudioPlayer features={visibleAudioFeatures} zoom={state.zoom} />;
        return (
            <div>
                {audioPlayer}
                {state.uis.map((p, i) => <InfoVisualizer key={i} uiState={p} zoom={state.zoom} />)}
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

