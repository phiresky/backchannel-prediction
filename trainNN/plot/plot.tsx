import { Line } from 'react-chartjs-2';
import * as React from 'react';
import * as ReactDOM from 'react-dom';

interface SingleEvalResult {
    "selected": number,
    "relevant": number,
    "true_positives": number,
    "false_positives": number,
    "false_negatives": number,
    "precision": number,
    "recall": number,
    "f1_score": number
}
interface EvalResult {
    config: {
        margin_of_error: [number, number]
    },
    totals: SingleEvalResult,
    details: { [convid: string]: SingleEvalResult }
}
const config = (version: string) => `../out/${version}/config.json`;
const evalResult = (version: string) => `../../evaluate/out/${version}/results.json`;
const all = [
    "v026-sgd-1",
    "v027-momentum-1",
    "v028-nesterov-1",
    "v029-adadelta-1",
    "v030-sgd-1-init0"
];
const titles = {
    "v026-sgd-1": "sgd, learning rate=1",
    "v027-momentum-1": "momentum, learning rate=1",
    "v028-nesterov-1": "nesterov, learning rate=1",
    "v029-adadelta-1": "adadelta, learning rate=1"
} as { [version: string]: string };
const relevant = [
    ...all.map(version => ({ version })),
];

const useMinMax = location.search.includes("fixed");

type VGProps = { evalInfo?: EvalResult[], version: string, data: any, options: any };

class VersionGUI extends React.Component<VGProps, {}> {
    render() {
        const {evalInfo, version, data, options} = this.props;
        if(evalInfo)
            var [bestIndex, bestValue] = evalInfo.map(res => res.totals.f1_score)
                .reduce(([maxInx, max], cur, curInx) => cur > max ? [curInx, cur] : [maxInx, max], [-1, -Infinity]);
        return (
            <div key={version}>
                <h3>{version in titles ? `${titles[version]} (${version})` : `${version}`}</h3>
                <Line data={data} options={options} />
                {evalInfo &&
                    <table className="evalTable">
                        <tr><th /><th /><th>Precision</th><th>Recall</th><th>F1</th></tr>
                        {evalInfo.map((v, k) =>
                            <tr className={k === bestIndex ? "highlighted" : ""}>
                                {v.config.margin_of_error.map(x => <th>{x}s</th>)}
                                <td>{v.totals.precision.toFixed(3)}</td>
                                <td>{v.totals.recall.toFixed(3)}</td>
                                <td>{v.totals.f1_score.toFixed(3)}</td>
                            </tr>
                        )}
                    </table>
                }
            </div>
        );
    }
}
class GUI extends React.Component<{}, { results: VGProps[] }> {
    constructor() {
        super();
        this.state = { results: [] };
        this.retrieveData();
    }
    async retrieveData() {
        const versions = await fetch("dist/dirindex.txt").then(resp => resp.text());
        const relevant = versions.split("\n").map(version => ({ version }));
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
        for (const {version} of relevant) {
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
            const results = this.state.results.slice();
            results.push({
                version,
                evalInfo,
                data: {
                    datasets: plotData
                },
                options: {
                    scales: {
                        xAxes: [{
                            type: 'linear',
                            position: 'bottom'
                        }],
                        yAxes: [
                            { position: "left", id: "Loss", scaleLabel: { display: true, labelString: "Loss" }, ticks: useMinMax ? { min: 0.5, max: 0.68 } : {} },
                            { position: "right", id: "Error", scaleLabel: { display: true, labelString: "Error" }, ticks: useMinMax ? { min: 0.2, max: 0.5 } : {} }
                        ]
                    }
                }
            });
            this.setState({ results });
        }
    }
    render() {

        return (
            <div className="gui">
                {this.state.results.map(info => <VersionGUI {...info} />)}
            </div>
        );
    }
}

document.addEventListener("DOMContentLoaded", () => {
    ReactDOM.render(<GUI />, document.getElementById("content"));
});