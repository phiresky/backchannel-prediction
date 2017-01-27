import { Line } from 'react-chartjs-2';
import * as React from 'react';
import * as ReactDOM from 'react-dom';
import { Table } from 'reactable';
import { observable, action, computed } from 'mobx';
import * as mobx from 'mobx';
import { observer, Observer } from 'mobx-react';
import 'react-select/dist/react-select.css';
import './style.css';
import * as Select from 'react-select';
import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
import * as util from './util';
import * as mu from 'mobx-utils';
import * as Rx from 'rxjs';

// defined by webpack
declare var VERSIONS: string[];

interface SingleEvalResult {
    "selected": number,
    "relevant": number,
    "true_positives": number,
    "false_positives": number,
    "false_negatives": number,
    "precision": number,
    "recall": number,
    "f1_score": number,
}
interface EvalResult {
    shown?: true,
    config: {
        margin_of_error: [number, number],
        min_talk_len: number | null,
        threshold: number,
        epoch: string,
        weights_file: string
    },
    totals: { eval: SingleEvalResult, valid: SingleEvalResult }
}
const path = (version: string, file: string) => `../../../trainNN/out/${version}/${file}`;
const evalResult = (version: string) => `../../../evaluate/out/${version}/results.json`;
const titles = {
    "v026-sgd-1": "sgd, learning rate=1",
    "v027-momentum-1": "momentum, learning rate=1",
    "v028-nesterov-1": "nesterov, learning rate=1",
    "v029-adadelta-1": "adadelta, learning rate=1"
} as { [version: string]: string };
const ignore = [
    "v036-online-lstm", "v036-online-lstm-dirty"
]

type VGPropsMaybe = { ok: false, error: string, version: string, log: string | null } | VGProps;
type VGProps = { ok: true, evalInfo?: EvalResult[], version: string, data: {datasets: {label: string, yAxisID: string, data: {x: number, y: number}[]}[]}, options: any };

function persist<T>({ initial, prefix = "roulette:" }: { initial: T, prefix?: string }) {
    return (prototype: Object, name: string) => {
        const stored = localStorage.getItem(prefix + name);
        const value = observable(stored === null ? initial : JSON.parse(stored));
        Object.defineProperty(prototype, name, {
            get: () => value.get(),
            set: (v) => {
                value.set(v);
                localStorage.setItem(prefix + name, JSON.stringify(v));
            }
        });
    };
}

function StringEnum<T extends string>(...values: T[]): {[x in T]: x} {
    return Object.assign({}, ...values.map(value => ({ [value as any]: value })));
}
const xextractors: { [xaxis: string]: (r: EvalResult) => number } = {
    "Epoch": r => +r.config.epoch,
    "Margin of Error Center": r => r.config.margin_of_error.reduce((a, b) => (a + b) / 2),
    "Margin of Error Width": r => r.config.margin_of_error.reduce((a, b) => +(b - a).toFixed(3)),
    "Min Talk Len": r => r.config.min_talk_len === null ? -1 : r.config.min_talk_len,
    "Threshold": r => r.config.threshold,
};
const yextractors: { [yaxis: string]: (r: EvalResult) => number } = {
    "Valid: Precision": r => r.totals.valid.precision,
    "Valid: Recall": r => r.totals.valid.recall,
    "Valid: F1 Score": r => r.totals.valid.f1_score,
    "Eval: Precision": r => r.totals.eval.precision,
    "Eval: Recall": r => r.totals.eval.recall,
    "Eval: F1 Score": r => r.totals.eval.f1_score
}
const colors = "#3366CC,#DC3912,#FF9900,#109618,#990099,#3B3EAC,#0099C6,#DD4477,#66AA00,#B82E2E,#316395,#994499,#22AA99,#AAAA11,#6633CC,#E67300,#8B0707,#329262,#5574A6,#3B3EAC".split(",");
const toPrecision = (x: number | null) => typeof x === "number" ? x.toPrecision(3) : NaN;
const toTableData = mobx.createTransformer(function toTableData(v: EvalResult) {
    return {
        ...Object.assign({}, ...Object.keys(xextractors).map(name => ({ [name]: toPrecision(xextractors[name](v)) }))),
        ...Object.assign({}, ...Object.keys(yextractors).map(name => ({ [name]: toPrecision(yextractors[name](v)) }))),
    }
});
const toJS = mobx.createTransformer(x => mobx.toJS(x));
@observer
class VersionEvalDetailGUI extends React.Component<VGProps, {}> {
    validXaxes = Object.keys(xextractors);
    @observable config = {
        xaxis: "Margin of Error Center",
        yaxes: ["Valid: F1 Score", "Valid: Precision", "Valid: Recall"]
    }
    render() {
        const { xaxis, yaxes } = this.config;
        if (!this.props.evalInfo) return <div>no evaluation data</div>;
        const options = {
            scales: {
                xAxes: [{
                    type: 'linear',
                    position: 'bottom',
                    scaleLabel: {
                        visible: true,
                        labelString: xaxis
                    }
                }],
                yAxes: [
                    { position: "left", id: "rating", ticks: { min: 0 } },
                    /*{ position: "right", id: "Error", scaleLabel: { display: true, labelString: "Error" }, ticks: {} }*/
                ]
            }
        };
        const _ = this.validXaxes; type XAxis = keyof typeof _;
        const map = new Map<string, EvalResult[]>();

        const extractor = xextractors[xaxis];
        const { [xaxis]: mine, ...otherso } = xextractors;
        const others = Object.keys(otherso).sort().map(x => xextractors[x]);
        for (const evalInfo of this.props.evalInfo) {
            const myValue = mine(evalInfo);
            const otherValues = others.map(extractor => extractor(evalInfo)).join(",");
            if (!map.has(otherValues)) map.set(otherValues, []);
            map.get(otherValues)!.push(evalInfo);
        }
        const relevants = Array.from(map.values()).reduce((a, b) => a.length > b.length ? a : b);
        console.log(map);
        const datasets = yaxes.map((yaxis, i) => {
            const data = relevants.map(ele => ({ x: extractor(ele), y: yextractors[yaxis](ele) }));
            data.sort((a, b) => a.x - b.x);
            console.log(data);
            return {
                label: yaxis,
                borderColor: colors[i],
                fill: false,
                yAxisID: "rating",
                lineTension: 0.2,
                pointRadius: 2,
                data
            };
        });
        console.log(yaxes);
        return (
            <div>
                x-axis: <Select searchable={false} clearable={false}
                    value={xaxis}
                    options={Object.keys(xextractors).map(value => ({ value, label: value }))}
                    onChange={v => this.config.xaxis = (v as any).value} />
                y-axis: <Select searchable={false} clearable={false}
                    value={toJS(yaxes)}
                    options={Object.keys(yextractors).map(value => ({ value, label: value }))}
                    multi
                    onChange={x => this.config.yaxes = (x as any[]).map(x => x.value)} />
                <Line data={{ datasets }} options={options} />
                <Table className="evalTable" sortable
                    itemsPerPage={6}
                    filterable={"Margin of Error Width,Threshold,Min Talk Len".split(",")}
                    data={relevants.map(v => toTableData(v))}
                />
            </div>
        )
    }
}


@observer
class VersionEpochsGUI extends React.Component<VGProps, {}> {
    render() {
        const { evalInfo, version, data, options } = this.props;
        /*if(evalInfo)
            var [bestIndex, bestValue] = evalInfo.map(res => res.totals.eval !== null ? res.totals.eval.precision : res.totals.precision)
                .reduce(([maxInx, max], cur, curInx) => cur > max ? [curInx, cur] : [maxInx, max], [-1, -Infinity]);*/
        const isNewerVersion = +version.slice(1, 4) >= 42;
        if (isNewerVersion) {
            var defaultSort = { column: 'Valid: F1 Score', direction: 'desc' };
        } else {
            var defaultSort = { column: 'Eval: F1 Score', direction: 'desc' };
        }
        return (
            <div>
                <Line data={toJS(data)} options={toJS(options)} />
                {evalInfo &&
                    <div>
                        Eval Results for best epoch according to val_error ({evalInfo[0].config.weights_file}):
                    <Table className="evalTable" sortable
                            defaultSort={defaultSort} itemsPerPage={6}
                            filterable={"Margin of Error Width,Threshold,Min Talk Len".split(",")}
                            data={evalInfo.map(v => toTableData(v))}
                        />
                    </div>
                }
            </div>
        );
    }
}

const logStyle = { maxHeight: "400px", overflow: "scroll" };
function LogGui(p: VGProps) {
    let res = observable({ txt: "Loading..." });
    (async () => {
        res.txt = await (await fetch(path(p.version, "train.log"))).text();
    })();
    return <Observer>{() => <pre style={logStyle}>{res.txt}</pre>}</Observer>;
}
@observer
class VersionGUI extends React.Component<{ gui: GUI, p: VGPropsMaybe }, {}> {
    @observable tab = 0;
    render() {
        const p = this.props.p;
        const version = p.version;
        const isNewerVersion = +version.slice(1, 4) >= 42;
        if (isNewerVersion) {
            var [gitversion, title] = version.split(":");
        } else {
            var gitversion = version;
            var title = titles[version];
        }
        if (p.ok) {
            const { evalInfo } = p;
            var inner = (
                <Tabs onSelect={i => this.tab = i} selectedIndex={this.tab}>
                    <TabList>
                        <Tab>Training Graph</Tab>
                        <Tab>Training Log</Tab>
                        {evalInfo && <Tab>Eval Detail Graphs ({evalInfo.length} evaluations) </Tab>}
                    </TabList>
                    <TabPanel><VersionEpochsGUI {...p} /></TabPanel>
                    <TabPanel><LogGui {...p} /></TabPanel>
                    {evalInfo && <TabPanel><VersionEvalDetailGUI {...p} /></TabPanel>}
                </Tabs>
            );
        } else {
            inner = (
                <div><p>Error: {p.error}</p>
                    {p.log ? <div>Log: <pre style={logStyle}>{p.log}</pre></div> : <div>(Log file could not be loaded)</div>}
                </div>
            );
        }
        return (
            <div>
                <button style={{ float: "right" }} onClick={() => this.props.gui.load(version)}>reload</button>
                <h3>{title ? `${title}` : `${version}`}</h3>
                <p>
                    Git version: {gitversion}
                    <a href={path(version, "config.json")}>Complete configuration json</a>
                    <a href={path(version, "train.log")}>Training log</a>
                    {isNewerVersion || <a href={path(version, "network_model.py")}>Network model</a>}
                    {p.ok && p.evalInfo && <a href={evalResult(version)}>Eval Result json</a>}
                </p>
                {inner}
            </div>
        );
    }
}

const axis_keys = [{
    key: "training_loss",
    color: "blue",
    axis: "Loss"
}, {
    key: "validation_loss",
    color: "red",
    axis: "Loss"
}, {
    key: "validation_error",
    color: "green",
    axis: "Error"
}];
function maxByKey<T>(data: T[], extractor: (t: T) => number) {
    return data.reduce((curMax, ele) => extractor(curMax) > extractor(ele) ? curMax : ele);
}

@observer class OverviewStats extends React.Component<{results: VGPropsMaybe[]}, {}> {
    bestResult = mobx.createTransformer((data: EvalResult[]) => {
        return maxByKey(data, info => info.totals.valid.f1_score);
    });
    render() {
        const results = this.props.results.filter(res => res.ok && res.evalInfo).map(res => ({version: res.version, info: this.bestResult((res as VGProps).evalInfo!)}));
        return (
            <div>
                <h4>Best for each</h4>
                <Table className="evalTable"
                    defaultSort={{ column: 'Valid: F1 Score', direction: 'desc' }}
                    data={results.map(result => ({Git:result.version.split(":")[0], Version: result.version.split(":")[1], ...toTableData(result.info)}))}
                    sortable filterable/>
            </div>
        );
    }
}
@observer
class GUI extends React.Component<{}, {}> {
    constructor() {
        super();
        this.retrieveData();
    }
    @observable useMinMax = false;
    @observable showUnevaluated = true;
    @observable onlyNew = true;
    @observable results = observable.map() as any as Map<string, VGPropsMaybe>;
    @observable loaded = 0; @observable total = 1;
    axisMinMax(results: VGPropsMaybe[], filter: string, axisId: string) {
        const arr = results.filter(result => this.resultVisible(result, filter))
            .map(arr => !arr.ok ? []: arr.data.datasets.filter(dataset => dataset.yAxisID === axisId).map(dataset => dataset.data.map(data => data.y)));
        const data = arr.reduce((a, b) => a.concat(b.reduce((a, b) => a.concat(b)), [] as number[]), [] as number[]);
        const min = Math.min(...data);
        const max = Math.max(...data);
        return {min, max};
    }
    options(results: VGPropsMaybe[], filter: string) {
        return {
            scales: {
                xAxes: [{
                    type: 'linear',
                    position: 'bottom'
                }],
                yAxes: [
                    { position: "left", id: "Loss", scaleLabel: { display: true, labelString: "Loss" }, ticks: this.useMinMax ? this.axisMinMax(results, filter, "Loss") : {} },
                    { position: "right", id: "Error", scaleLabel: { display: true, labelString: "Error" }, ticks: this.useMinMax ? this.axisMinMax(results, filter, "Error") : {} }
                ],
            },
            animation: { duration: 0 }
        };
    }
    async errorTryGetLog(version: string, error: string): Promise<VGPropsMaybe> {
        const resp = await fetch(path(version, "train.log"));
        let log = null;
        if (resp.ok) {
            log = await resp.text();
        }
        return { ok: false, version, error, log };
    }
    async load(version: string) {
        const res = await this.retrieveDataFor(version);
        if (res) this.results.set(version, res);
    }
    async retrieveDataFor(version: string): Promise<VGPropsMaybe | null> {
        const resp = await fetch(path(version, "config.json"));
        if (!resp.ok) {
            return await this.errorTryGetLog(version, `${path(version, "config.json")} could not be found`);
        }
        let data;
        try {
            data = await resp.json();
        } catch (e) {
            return await this.errorTryGetLog(version, `Error parsing ${version}/config.json: ${e}`);
        }
        const evalResp = await fetch(evalResult(version));
        let evalInfo: EvalResult[] | undefined;
        if (evalResp.ok) evalInfo = await (evalResp.json());
        if (evalInfo) {
            // upgrade
            evalInfo = evalInfo.map(x => x.totals.eval ? x : { ...x, totals: { eval: x.totals, valid: {} } } as any);
            evalInfo.forEach(i => i.totals.valid.f1_score === 1 && (i.totals.valid.f1_score = 0));
        }
        const stats = data.train_output.stats;

        if (!stats["0"]["validation_loss"]) return null;
        const plotData = axis_keys.map((info, i) => ({
            label: info.key,
            borderColor: colors[i],
            fill: false,
            yAxisID: info.axis,
            lineTension: 0.2,
            pointRadius: 2,
            data: Object.entries(stats).map(([x, stat]) => ({
                x: +x,
                y: stat[info.key]
            }))
        }));
        return {
            ok: true,
            version,
            evalInfo,
            data: {
                datasets: plotData
            },
            options: {}
        };
    }
    async retrieveData() {
        const relevant = VERSIONS.filter(version => !ignore.includes(version)).map(version => ({ version }));
        this.total = relevant.length + 1;
        const promi: Promise<any>[] = [];
        for (const { version } of relevant) {
            promi.push(this.load(version).then(() => this.loaded++));
        }
        await Promise.all(promi);
        this.loaded = this.total;
    }
    @persist({ initial: ".*" }) filter: string;
    getFilter = () => this.filter;
    getResults = () => this.results;
    resultVisible(result: VGPropsMaybe, filter: string) {
        let results = [result];
        if (!this.showUnevaluated) results = results.filter(res => res.ok && res.evalInfo);
        if (this.onlyNew) results = results.filter(res => res.version.indexOf("unified") >= 0);
        try {
            const fltr = RegExp(filter);
            results = results.filter(res => res.version.search(fltr) >= 0);
        } catch (e) {
            console.log("invalid regex", filter);
            return false;
        }
        return !!results[0];
    }
    render() {
        const throttledFilter = util.throttleGet(500, this.getFilter);
        let results = Array.from(util.throttleGet(500, this.getResults).values());
        results = results.sort((a, b) => a.version.localeCompare(b.version));
        results = results.filter(result => this.resultVisible(result, throttledFilter));
        const options = this.options(results, throttledFilter);
        return (
            <div>
                <div>
                    <label>Show unevaluated:
                        <input type="checkbox" checked={this.showUnevaluated} onChange={x => this.showUnevaluated = x.currentTarget.checked}
                        />
                    </label>
                    <label>Use fixed min/max:
                        <input type="checkbox" checked={this.useMinMax} onChange={x => this.useMinMax = x.currentTarget.checked}
                        />
                    </label>
                    <label>Only unified:
                        <input type="checkbox" checked={this.onlyNew} onChange={x => this.onlyNew = x.currentTarget.checked}
                        />
                    </label>
                </div>
                <label>Filter:
                    <Observer>
                        {() => <input value={this.filter} onChange={e => this.filter = e.currentTarget.value} />}
                    </Observer>
                </label>
                <Observer>
                    {() => this.loaded < this.total ? <h3>Loading ({this.loaded}/{this.total})...</h3> : <h3>Loading complete</h3>}
                </Observer>
                <OverviewStats results={results} />
                <div className="gui">
                    {results.map(info => <VersionGUI key={info.version} gui={this} p={{ ...info, options }} />)}
                </div>
            </div>
        );
    }
}
export let gui: any;
document.addEventListener("DOMContentLoaded", () => {
    const div = document.createElement("div");
    document.body.appendChild(div);
    gui = ReactDOM.render(<GUI />, div);
});

declare var module: any;
Object.assign(window, { plot: module.exports, mu, Rx, mobx })