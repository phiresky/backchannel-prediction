import * as c from './client';
import * as React from 'react';
import {observer} from 'mobx-react';
import * as mobx from 'mobx';
import * as util from './util';
import * as Data from './Data';

function renderWaveform(ctx: CanvasRenderingContext2D, y: number, w: number, h: number,
        givenRange: [number, number]|null, config: c.VisualizerConfig, data: Data.DataIterator, zoom: {left: number, right: number}) {
    const start = Math.floor(zoom.left * data.iterator.count);
    const end = Math.floor(zoom.right * data.iterator.count);
    const length = end - start;
    const display = util.getMinMax(givenRange, config, data, Math.max(start, 0), Math.min(end, data.iterator.count));

    const displayMinMax = display.max - display.min;
    let lasth1 = NaN, lasth2 = NaN;
    ctx.fillStyle = c.globalConfig.maxColor;
    const rmsc = new Array<number>(2 * w);
    for (let x = 0; x < w; x++) {
        const from = x / w, to = (x + 1) / w;
        const fromSample = start + Math.floor(length * from);
        const toSample = start + Math.ceil(length * to);
        if(fromSample < 0) continue;
        if(toSample >= data.iterator.count) continue;
        const {min, max, rms2:rmsSq, sum, count} = data.data.stats(data.iterator, fromSample, toSample);
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
    ctx.fillStyle = c.globalConfig.rmsColor;
    for (let x = 0; x < w; x++) {
        ctx.fillRect(x, y + rmsc[2*x], 1, rmsc[2*x + 1] - rmsc[2*x]);
    }
    return display;
}

function renderDarkness(ctx: CanvasRenderingContext2D, y: number, w: number, h: number,
        givenRange: [number, number]|null, config: c.VisualizerConfig, data: Data.DataIterator, zoom: {left: number, right: number}) {
    const start = Math.floor(zoom.left * data.iterator.count);
    const end = Math.floor(zoom.right * data.iterator.count);
    const length = end - start;
    const display = util.getMinMax(givenRange, config, data, start, end);
    for (let x = 0; x < w; x++) {
        const from = x / w, to = (x + 1) / w;
        const fromSample = start + Math.floor(length * from);
        const toSample = start + Math.ceil(length * to);
        const {min, max, rms2, sum, count} = data.data.stats(data.iterator, fromSample, toSample);
        const avg = sum / count;
        ctx.fillStyle = `rgba(0,0,0,${(avg - display.min)/(display.max - display.min)})`;
        ctx.fillRect(x, y, 1, h);
    }
    return display;
}

abstract class CanvasRenderer<P> extends React.Component<c.VisualizerProps<P>, {}> {
    canvas: HTMLCanvasElement;
    disposers: (() => void)[] = [];
    abstract preferredHeight: number;
    abstract renderCanvas(canvas: HTMLCanvasElement): void;
    canvasWidthFactor = 1;
    filter?: string;
    @mobx.observable
    canvasZoom = mobx.asStructure({
        left: 10,
        right: 10 + this.canvasWidthFactor
    })
    render() {
        const zoom = this.props.gui.zoom;
        const canvasZoom = this.canvasZoom;
        const zw = zoom.right - zoom.left;
        const cw = canvasZoom.right - canvasZoom.left;
        if(!(canvasZoom.left <= zoom.left && zoom.right <= canvasZoom.right) || Math.abs(this.canvasWidthFactor * zw / cw - 1) > 0.01) {
            // const anchor = (zoom.left + zoom.right) / 2;
            // prerender to the right, offset a random bit to prevent multiple tracks from rendering on the same frame
            const anchor = zoom.left + (zoom.right - zoom.left) * 0.1 * Math.random();
            mobx.runInAction("canvas zoom", () => 
            {
                Object.assign(this.canvasZoom, util.rescale(zoom, this.canvasWidthFactor, anchor));
            });
        }
        const screenW = this.props.gui.width;
        const canvW = canvasZoom.right - canvasZoom.left;
        const screenLeft = (zoom.left - canvasZoom.left) / (zoom.left - zoom.right) * screenW;
        const screenRight = screenLeft + this.canvasWidthFactor * canvW * screenW;
        const border = 1;
        return <div style={{height:"100%", position: "relative", borderStyle:"solid", borderWidth:border+"px", overflow: "hidden"}}>
            <canvas style={{display:"block", filter: this.filter, position: "absolute", width: Math.round(this.canvasWidthFactor * screenW)+"px", left: screenLeft, height: "100%",}}
                ref={c => this.canvas = c}/>
        </div>;
    }
    componentDidMount() {
        this.disposers.push(mobx.autorun("renderCanvas", () => this.renderCanvas(this.canvas)));
    }
    componentWillUnmount() {
        for(const disposer of this.disposers) disposer();
    }
}

abstract class MultiCanvasRenderer extends CanvasRenderer<c.NumFeature> {
    abstract singleRenderFunction: 
        (ctx: CanvasRenderingContext2D, y: number, w: number, h: number,
            givenRange: [number, number]|null, config: c.VisualizerConfig, data: Data.DataIterator, zoom: {left: number, right: number})
            => {min: number, max: number};

    @mobx.action
    setCurrentRange(range: {min: number, max: number}) {
        this.props.uiState.currentRange = range;
    }

    renderCanvas(canvas: HTMLCanvasElement) {
        const target = this.canvas;
        target.width = Math.round(this.props.gui.width * this.canvasWidthFactor);
        target.height = target.clientHeight;
        const data = this.props.feature.data;
        const w = target.width;
        const h = target.height;
        const ctx = target.getContext("2d")!;
        const dim = data.shape[1];
        let range;
        for(let y = 0; y < dim; y++) {
            range = this.singleRenderFunction(ctx, Math.floor(y / dim * h), w, Math.floor(h / dim),
                this.props.feature.range, this.props.uiState.config, {data, iterator:data.iterate("ALL", y)}, this.canvasZoom)//this.props.gui.zoom);
        }
        if(range) this.setCurrentRange(range);
    }
}
@observer
export class AudioWaveform extends MultiCanvasRenderer {
    preferredHeight = 100;
    filter = "drop-shadow(rgba(0, 0, 0, 0.2) 5px 5px 5px)";
    singleRenderFunction = renderWaveform;
}
@observer
export class Darkness extends MultiCanvasRenderer {
    preferredHeight = 50;
    singleRenderFunction = renderDarkness;
}