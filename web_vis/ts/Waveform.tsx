import {Feature, NumFeature, NumFeatureFMatrix, NumFeatureSVector, Highlights, 
        Visualizer, VisualizerProps, VisualizerConfig, globalConfig} from './client';
import * as React from 'react';
import {observer} from 'mobx-react';
import {observable, autorun, computed, action} from 'mobx';
import * as util from './util';

function renderWaveform(ctx: CanvasRenderingContext2D, y: number, w: number, h: number,
        givenRange: [number, number]|null, config: VisualizerConfig, data: ArrayLike<number>, zoom: {left: number, right: number}) {
    const start = Math.floor(zoom.left * data.length);
    const end = Math.floor(zoom.right * data.length);
    const length = end - start;
    const display = util.getMinMax(givenRange, config, data, start, end);

    const displayMinMax = display.max - display.min;
    let lasth1 = NaN, lasth2 = NaN;
    ctx.fillStyle = globalConfig.maxColor;
    const rmsc = new Array<number>(2 * w);
    for (let x = 0; x < w; x++) {
        const from = x / w, to = (x + 1) / w;
        const fromSample = start + Math.floor(length * from);
        const toSample = start + Math.ceil(length * to);
        const {min, max, rms2:rmsSq, sum, count} = util.stats(data, fromSample, toSample);
        const avg = sum / count;
        let h1 = Math.round(h * (1 - (max - display.min)/displayMinMax));
        let h2 = Math.round(h * (1 - (min - display.min)/displayMinMax));
        const hAvg = Math.round(h * (1 - (avg - display.min)/displayMinMax));
        if (x > 0) {
            if (h1 > lasth2) h1 = lasth2 + 1;
            if (h2 < lasth1) h2 = lasth1 - 1;
        }
        if(h1 >= h2) h2 = h1 + 1;
        lasth1 = h1;
        lasth2 = h2;
        ctx.fillRect(x, y + h1, 1, h2 - h1);
        const rms = Math.sqrt(rmsSq * Math.sqrt(count) / 4);
        let rms1 = Math.round((1 - rms / displayMinMax) * hAvg);
        let rms2 = Math.round((1 + rms / displayMinMax) * hAvg);
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
    return display;
}

function renderDarkness(ctx: CanvasRenderingContext2D, y: number, w: number, h: number,
        givenRange: [number, number]|null, config: VisualizerConfig, data: ArrayLike<number>, zoom: {left: number, right: number}) {
    const start = Math.floor(zoom.left * data.length);
    const end = Math.floor(zoom.right * data.length);
    const length = end - start;
    const display = util.getMinMax(givenRange, config, data, start, end);
    for (let x = 0; x < w; x++) {
        const from = x / w, to = (x + 1) / w;
        const fromSample = start + Math.floor(length * from);
        const toSample = start + Math.ceil(length * to);
        const {min, max, rms2, sum, count} = util.stats(data, fromSample, toSample);
        const avg = sum / count;
        ctx.fillStyle = `rgba(0,0,0,${(avg - display.min)/(display.max - display.min)})`;
        ctx.fillRect(x, y, 1, h);
    }
    return display;
}

abstract class CanvasRenderer<P> extends React.Component<VisualizerProps<P>, {}> {
    canvas: HTMLCanvasElement;
    disposers: (() => void)[] = [];

    abstract renderCanvas(canvas: HTMLCanvasElement): void;

    render() {
        const border = 1;
        return <div>
            <canvas style={{width: "100%", height: globalConfig.visualizerHeight+"px", borderStyle:"solid", borderWidth:border+"px"}}
                ref={c => this.canvas = c}/>
        </div>;
    }
    componentDidUpdate() {
        this.renderCanvas(this.canvas);
    }
    componentDidMount() {
        this.disposers.push(autorun("renderCanvas", () => this.renderCanvas(this.canvas)));
    }
    componentWillUnmount() {
        for(const disposer of this.disposers) disposer();
    }
}

abstract class MultiCanvasRenderer extends CanvasRenderer<NumFeature> {
    data: ArrayLike<number>[];

    abstract singleRenderFunction: 
        (ctx: CanvasRenderingContext2D, y: number, w: number, h: number,
            givenRange: [number, number]|null, config: VisualizerConfig, data: ArrayLike<number>, zoom: {left: number, right: number})
            => {min: number, max: number};
     
    constructor(props: VisualizerProps<NumFeature>) {
        super(props);
        if(props.feature.typ === "FeatureType.SVector") this.data = [props.feature.data];
        else if(props.feature.typ === "FeatureType.FMatrix") {
            const data = props.feature.data;
            this.data = props.feature.data[0].map((_,i) => data.map(v => v[i]));
        } else throw Error("unknown type "+(props.feature as any).typ);
    }

    @action
    setCurrentRange(range: {min: number, max: number}) {
        this.props.uiState.currentRange = range;
    }

    renderCanvas(canvas: HTMLCanvasElement) {
        const target = this.canvas;
        this.props.gui.windowWidth; // for mobx tracking
        target.width = target.clientWidth;
        target.height = target.clientHeight;
        
        const w = target.width;
        const h = target.height - 1;
        const ctx = target.getContext("2d")!;
        const dim = this.data.length;
        let range;
        for(let y = 0; y < dim; y++) {
            range = this.singleRenderFunction(ctx, Math.floor(y / dim * h), w, Math.floor(h / dim), this.props.feature.range, this.props.uiState.config, this.data[y], this.props.gui.zoom);
        }
        if(range) this.setCurrentRange(range);
    }
}
@observer
export class AudioWaveform extends MultiCanvasRenderer {
    singleRenderFunction = renderWaveform;
}
@observer
export class Darkness extends MultiCanvasRenderer {
    singleRenderFunction = renderDarkness;
}