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
    const minMax = config.max - config.min;
    let lasth1 = NaN, lasth2 = NaN;
    ctx.fillStyle = globalConfig.maxColor;
    const rmsc = new Array<number>(2 * w);
    for (let x = 0; x < w; x++) {
        const from = x / w, to = (x + 1) / w;
        const fromSample = start + Math.floor(length * from);
        const toSample = start + Math.ceil(length * to);
        const {min, max, rms2:rmsSq, sum, count} = util.stats(data, fromSample, toSample);
        const avg = sum / count;
        let h1 = Math.round(h * (1 - (max - config.min)/minMax));
        let h2 = Math.round(h * (1 - (min - config.min)/minMax));
        const hAvg = Math.round(h * (1 - (avg - config.min)/minMax));
        if (x > 0) {
            if (h1 > lasth2) h1 = lasth2 + 1;
            if (h2 < lasth1) h2 = lasth1 - 1;
        }
        if(h1 >= h2) h2 = h1 + 1;
        lasth1 = h1;
        lasth2 = h2;
        ctx.fillRect(x, y + h1, 1, h2 - h1);
        const rms = Math.sqrt(rmsSq * Math.sqrt(count) / 4);
        let rms1 = Math.round((1 - rms / minMax) * hAvg);
        let rms2 = Math.round((1 + rms / minMax) * hAvg);
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
    disposers: (() => void)[] = [];

    constructor(props: VisualizerProps<NumFeatureSVector>) {
        super(props);
        if(props.feature.typ !== "FeatureType.SVector") throw Error("not svec");
        window.addEventListener("resize", event => this.forceUpdate());
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