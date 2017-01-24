import { Line } from 'react-chartjs-2';
import * as React from 'react';
import * as ReactDOM from 'react-dom';
import { Table } from 'reactable';
import { toJS, observable, action, computed } from 'mobx';
import { observer } from 'mobx-react';

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
const config = (version: string) => `../../trainNN/out/${version}/config.json`;
const log = (version: string) => `../../trainNN/out/${version}/train.log`;
const model = (version: string) => `../../trainNN/out/${version}/network_model.py`;
const evalResult = (version: string) => `../../evaluate/out/${version}/results.json`;
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

@observer
class VersionGUI extends React.Component<VGProps, {}> {
    render() {
        const { evalInfo, version, data, options } = this.props;
        /*if(evalInfo)
            var [bestIndex, bestValue] = evalInfo.map(res => res.totals.eval !== null ? res.totals.eval.precision : res.totals.precision)
                .reduce(([maxInx, max], cur, curInx) => cur > max ? [curInx, cur] : [maxInx, max], [-1, -Infinity]);*/
        return (
            <div key={version}>
                <h3>{version in titles ? `${titles[version]} (${version})` : `${version}`}</h3>
                <Line key={Math.random()} data={toJS(data)} options={options} />
                <p><a href={log(version)}>Training log</a><a href={model(version)}>Network model</a></p>
                {evalInfo &&
                    <div>
                        Eval Results for best epoch according to val_error ({evalInfo[0].config.weights_file}):
                    <Table className="evalTable" sortable
                    /*defaultSort={[{column: 'Valid:F1 Score', direction: 'desc'}]}*/ itemsPerPage={6} pageButtonLimit={1}
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
            const resp = await fetch(config(version));
            if (!resp.ok) continue;
            const data = await resp.json();
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
            console.log(plotData);
            this.results.push({
                version,
                evalInfo,
                data: {
                    datasets: plotData
                },
                options: {}
            });
        }
    }
    render() {
        let results = this.results;
        if (!this.showUnevaluated) results = results.filter(res => res.evalInfo);
        if (this.onlyNew) results = results.filter(res => res.version.indexOf("unified") >= 0);
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
                <div className="gui">
                    {results.map(info => <VersionGUI key={info.version} {...info} options={this.options} />)}
                </div>
            </div>
        );
    }
}

document.addEventListener("DOMContentLoaded", () => {
    ReactDOM.render(<GUI />, document.getElementById("content"));
});