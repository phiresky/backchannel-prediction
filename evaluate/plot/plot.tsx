import { Line } from 'react-chartjs-2';
import * as React from 'react';
import * as ReactDOM from 'react-dom';
import { Table } from 'reactable';
import { toJS, observable, action, computed } from 'mobx';
import { observer, Observer } from 'mobx-react';
import './style.css';

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
    eval: null
}
interface EvalResult {
    config: {
        margin_of_error: [number, number],
        min_talk_len: number | null,
        threshold: number,
        epoch: string,
        weights_file: string
    },
    totals: SingleEvalResult | { eval: SingleEvalResult, valid: SingleEvalResult },
    details: { [convid: string]: SingleEvalResult }
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

type VGProps = { evalInfo?: EvalResult[], version: string, data: any, options: any };

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

@observer
class VersionGUI extends React.Component<VGProps, {}> {
    render() {
        const { evalInfo, version, data, options } = this.props;
        /*if(evalInfo)
            var [bestIndex, bestValue] = evalInfo.map(res => res.totals.eval !== null ? res.totals.eval.precision : res.totals.precision)
                .reduce(([maxInx, max], cur, curInx) => cur > max ? [curInx, cur] : [maxInx, max], [-1, -Infinity]);*/
        const isNewerVersion = +version.slice(1, 4) >= 42;
        if (isNewerVersion) {
            var defaultSort = { column: 'Valid: F1 Score', direction: 'desc' };
            var [gitversion, title] = version.split(":");
        } else {
            var defaultSort = { column: 'Eval: F1 Score', direction: 'desc' };
            var gitversion = version;
            var title = titles[version];
        }
        return (
            <div key={version}>
                <h3>{title ? `${title}` : `${version}`}</h3>
                <p>Git version: {gitversion}</p>
                <Line key={Math.random()} data={toJS(data)} options={options} />
                <p>
                    <a href={path(version, "config.json")}>Complete configuration</a>
                    <a href={path(version, "train.log")}>Training log</a>
                    {isNewerVersion || <a href={path(version, "network_model.py")}>Network model</a>}
                </p>
                {evalInfo &&
                    <div>
                        Eval Results for best epoch according to val_error ({evalInfo[0].config.weights_file}):
                    <Table className="evalTable" sortable
                            defaultSort={defaultSort} itemsPerPage={6}
                            filterable={"Margin of Error,Threshold,Min Talk Len".split(",")}
                            data={
                                evalInfo.map((v, k) => {
                                    const evalTotals = (v.totals.eval ? v.totals.eval : v.totals) as SingleEvalResult;
                                    const validTotals = (v.totals.eval ? (v.totals as any).valid : undefined) as SingleEvalResult | undefined;
                                    return {
                                        "Margin of Error": v.config.margin_of_error.map(s => `${s}s`).join(", "),
                                        "Threshold": v.config.threshold,
                                        "Min Talk Len": v.config.min_talk_len,
                                        "Valid: Precision": validTotals && validTotals.precision.toFixed(3),
                                        "Valid: Recall": validTotals && validTotals.recall.toFixed(3),
                                        "Valid: F1 Score": validTotals && validTotals.f1_score.toFixed(3),
                                        "Eval: Precision": evalTotals.precision.toFixed(3),
                                        "Eval: Recall": evalTotals.recall.toFixed(3),
                                        "Eval: F1 Score": evalTotals.f1_score.toFixed(3),

                                    }
                                })
                            }
                        />
                    </div>
                }
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
    @observable results: VGProps[] = [];
    @observable isLoading = true;
    @persist({ initial: ".*" }) filter: string;
    @computed get options() {
        return {
            scales: {
                xAxes: [{
                    type: 'linear',
                    position: 'bottom'
                }],
                yAxes: [
                    { position: "left", id: "Loss", scaleLabel: { display: true, labelString: "Loss" }, ticks: this.useMinMax ? { min: 0.5, max: 0.68 } : {} },
                    { position: "right", id: "Error", scaleLabel: { display: true, labelString: "Error" }, ticks: this.useMinMax ? { min: 0.2, max: 0.5 } : {} }
                ]
            }
        };
    }
    async retrieveData() {
        const relevant = VERSIONS.filter(version => !ignore.includes(version)).map(version => ({ version }));
        const keys = [{
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
        for (const { version } of relevant) {
            const resp = await fetch(path(version, "config.json"));
            if (!resp.ok) continue;
            let data;
            try {
                data = await resp.json();
            } catch (e) {
                console.error("error parsing", version, "skipping:", e);
                continue;
            }
            const evalResp = await fetch(evalResult(version));
            let evalInfo: EvalResult[] | undefined;
            if (evalResp.ok) evalInfo = await (evalResp.json());
            const stats = data.train_output.stats;
            if (Object.keys(stats).length < 3) continue;

            if (!stats["0"]["validation_loss"]) continue;
            const plotData = keys.map(info => ({
                label: info.key,
                borderColor: info.color,
                fill: false,
                yAxisID: info.axis,
                lineTension: 0.2,
                pointRadius: 2,
                data: Object.entries(stats).map(([x, stat]) => ({
                    x: +x,
                    y: stat[info.key]
                }))
            }));
            this.results.push({
                version,
                evalInfo,
                data: {
                    datasets: plotData
                },
                options: {}
            });
        }
        this.isLoading = false;
    }
    render() {
        let results = this.results;
        if (!this.showUnevaluated) results = results.filter(res => res.evalInfo);
        if (this.onlyNew) results = results.filter(res => res.version.indexOf("unified") >= 0);
        try {
            const fltr = RegExp(this.filter);
            results = results.filter(res => res.version.search(fltr) >= 0);
        } catch (e) {
            console.log("invalid regex", this.filter);
        }
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
                    <input value={this.filter} onChange={e => this.filter = e.currentTarget.value} />
                </label>
                <Observer>
                    {() => this.isLoading ? <h3>Loading...</h3> : <h3>Loading complete</h3>}
                </Observer>
                <div className="gui">
                    {results.map(info => <VersionGUI key={info.version} {...info} options={this.options} />)}
                </div>
            </div>
        );
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const div = document.createElement("div");
    document.body.appendChild(div);
    ReactDOM.render(<GUI />, div);
});