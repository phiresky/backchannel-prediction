import {NumFeatureFMatrix, NumFeatureSVector, Visualizer, VisualizerProps, VisualizerConfig, globalConfig} from './client';
import * as React from 'react';
import {observer} from 'mobx-react';
import {observable, autorun, computed} from 'mobx';
import * as util from './util';
/*let canvas = document.createElement("canvas");
canvas.width = 1000;
canvas.height = 300;
renderWaveform(canvas, data)*/
function renderWaveform(ctx: CanvasRenderingContext2D, y: number, w: number, h: number,
        config: VisualizerConfig, data: number[], zoom: {left: number, right: number}) {
    const start = Math.floor(zoom.left * data.length);
    const end = Math.floor(zoom.right * data.length);
    const length = end - start;
    if (config == "normalize") config = util.stats(data, start, end);
    
    const mid = (h / 2) | 0;

    let lasth1 = NaN, lasth2 = NaN;
    ctx.fillStyle = globalConfig.maxColor;
    const rmsc = new Array<number>(2 * w);
    for (let x = 0; x < w; x++) {
        const from = x / w, to = (x + 1) / w;
        const fromSample = start + Math.floor(length * from);
        const toSample = start + Math.ceil(length * to);
        const {min, max, rms2:rmsSq} = util.stats(data, fromSample, toSample);
        let h1 = Math.round(h * (1 - (max - config.min)/(config.max - config.min)));
        let h2 = Math.round(h * (1 - (min - config.min)/(config.max - config.min)));
        if (x > 0) {
            if (h1 > lasth2) h1 = lasth2 + 1;
            if (h2 < lasth1) h2 = lasth1 - 1;
        }
        if(h1 >= h2) h2 = h1 + 1;
        lasth1 = h1;
        lasth2 = h2;
        ctx.fillRect(x, y + h1, 1, h2 - h1);
        const rms = Math.sqrt(rmsSq);
        let rms1 = Math.round((1 - rms / (config.max - config.min)) * (h2 + h1)/2);
        let rms2 = Math.round((1 + rms / (config.max - config.min)) * (h2 + h1)/2);
        if (rms1 < h1 + 1) rms1 = h1 + 1;
        if (rms2 > h2 - 1) rms2 = h2 - 1;
        if (rms1 > rms2) rms2 = rms1;
        rmsc[2*x] = rms1;
        rmsc[2*x+1] = rms2;
    }
    ctx.fillStyle = globalConfig.rmsColor;
    for (let x = 0; x < w; x++) {
        ctx.fillRect(x, y + rmsc[2*x], 1, rmsc[2*x + 1] - rmsc[2*x]);
    }
}

@observer
export class AudioWaveform extends React.Component<VisualizerProps<NumFeatureSVector>, {}> {
    canvas: HTMLCanvasElement;
    playerBar: HTMLDivElement;
    disposers: (() => void)[] = [];
    audio: AudioContext;
    audioBuffer: AudioBuffer;
    audioSource: AudioBufferSourceNode | null;
    position = 0;
    playing: boolean;
    startedAt: number;
    @computed get left() {
        return this.canvas.getBoundingClientRect().left;
    }
    constructor(props: VisualizerProps<NumFeatureSVector>) {
        super(props);
        if(props.feature.typ !== "FeatureType.SVector") throw Error("not svec");
        window.addEventListener("resize", event => this.forceUpdate());
        this.audio = new AudioContext();
        const data = this.props.feature.data;
        this.audioBuffer = this.audio.createBuffer(1, data.length, props.feature.samplingRate * 1000);
        const arr = Float32Array.from(data, v => v / 2 ** 15);
        this.audioBuffer.copyToChannel(arr, 0);
    }

    renderWaveform() {
        const target = this.canvas;
        target.width = target.clientWidth;
        target.height = target.clientHeight;
        
        const w = target.width;
        const h = target.height - 1;
        const ctx = target.getContext("2d")!;
        const data = this.props.feature.data;
        renderWaveform(ctx, 0, w, h, this.props.config, data, this.props.zoom);
    }
    updateBar = () => {
        if(!this.audioSource) return;
        this.position = (this.audio.currentTime - this.startedAt) / this.audioBuffer.duration;
        this.playerBar.style.left = util.getPixelFromPosition(this.position, this.left, this.canvas.width, this.props.zoom) + "px";
        if(this.playing) requestAnimationFrame(this.updateBar);
    }
    stopPlayback() {
        if(this.audioSource) {
            this.audioSource.stop();
            this.audioSource = null;
        }
    }

    componentDidUpdate() {
        this.renderWaveform();
    }
    componentDidMount() {
        this.disposers.push(autorun(() => this.playerBar.style.left = util.getPixelFromPosition(this.position, this.left, this.canvas.width, this.props.zoom) + "px"));
        this.disposers.push(autorun(() => this.renderWaveform()));
        this.canvas.addEventListener("click", event => {
            this.stopPlayback();
            this.position = util.getPositionFromPixel(event.clientX, this.left, this.canvas.width, this.props.zoom)!;
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
                if(this.audioSource) {
                    this.stopPlayback();
                } else {
                    this.audioSource = this.audio.createBufferSource();
                    this.audioSource.buffer = this.audioBuffer;
                    this.audioSource.connect(this.audio.destination);
                    this.audioSource.start(0, this.position * this.audioBuffer.duration);
                    this.startedAt = this.audio.currentTime - this.position * this.audioBuffer.duration;
                    this.audioSource.addEventListener("ended", () => this.playing = false);
                    this.playing = true;
                    requestAnimationFrame(this.updateBar);
                }
            }
        });
    }
    componentWillUnmount() {
        for(const disposer of this.disposers) disposer();
    }
    render() {
        return <div>
            <div ref={p => this.playerBar = p} style={{position: "fixed", width: "2px", height: "100vh", top:0, left:-10, backgroundColor:"gray"}} />
            <canvas style={{width: "100%", border:"1px solid black"}} ref={c => this.canvas = c}/>
        </div>;
    }
}

@observer
export class MultiWaveform extends React.Component<VisualizerProps<NumFeatureFMatrix>, {}> {
    canvas: HTMLCanvasElement;
    playerBar: HTMLDivElement;
    disposers: (() => void)[] = [];
    data: number[][];
    constructor(props: VisualizerProps<NumFeatureFMatrix>) {
        super(props);
        if(props.feature.typ !== "FeatureType.FMatrix") throw Error("not svec");
        window.addEventListener("resize", event => this.forceUpdate());
        const data = this.props.feature.data;
        this.data = props.feature.data[0].map((_,i) => data.map(v => v[i]));
    }

    renderWaveform() {
        const target = this.canvas;
        target.width = target.clientWidth;
        target.height = target.clientHeight;
        
        const w = target.width;
        const h = target.height - 1;
        const ctx = target.getContext("2d")!;
        const data = this.props.feature.data;
        const dim = data[0].length;
        for(let y = 0; y < dim; y++) {
            renderWaveform(ctx, Math.floor(y / dim * h), w, Math.floor(h / dim), this.props.config, this.data[y], this.props.zoom);
        }
    }

    componentDidUpdate() {
        this.renderWaveform();
    }
    componentDidMount() {
        this.disposers.push(autorun(() => this.renderWaveform()));
    }
    componentWillUnmount() {
        for(const disposer of this.disposers) disposer();
    }
    render() {
        return <div>
            <canvas style={{width: "100%", border:"1px solid black"}} ref={c => this.canvas = c}/>
        </div>;
    }
}