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
                    <select value={this.range} onChange={this.changeRange.bind(this)} >
                        {Object.keys(InfoVisualizer.ranges).map(k => <option key={k} value={k}>{k}</option>)}
                    </select>
                </div>
                <div style={{flexGrow: 1}}>
                    <Visualizer config={uiState.visualizerConfig} zoom={zoom} feature={features.get(uiState.feature)!} />
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

const socket = new WebSocket("ws://localhost:8765");

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
        return (
            <div>
                {state.uis.map((p, i) => <InfoVisualizer key={i} uiState={p} zoom={state.zoom} />)}
            </div>
        );
    }
}


const gui = ReactDOM.render(<GUI />, document.getElementById("root"));
Object.assign(window, {gui, state, features});
/*window.addEventListener("wheel", event => {
    if (event.deltaY > 0) canvas.width *= 1.1;
    else canvas.width *= 0.9;
    renderWaveform(canvas, data);
});*/

