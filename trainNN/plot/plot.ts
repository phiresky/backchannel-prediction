
const config = (version: string) => `../out/${version}/config.json`;
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
async function retrieveData() {
    const versions = await fetch("dist/dirindex.txt").then(resp => resp.text());
    const relevant = versions.split("\n").map(version => ({version}));
    const div = document.getElementById("content")!;
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
        const stats = data.train_output.stats;
        if(Object.keys(stats).length < 3) continue;

        if(!stats["0"]["validation_loss"]) continue;
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
        const div2 = document.createElement("div");
        const canvas = document.createElement("canvas");
        if(version in titles) {
            var title = `${titles[version]} (${version})`;
        } else {
            title = `${version}`;
        }
        div2.appendChild(document.createTextNode(title));
        div2.appendChild(canvas);
        div.appendChild(div2);
        const myChart = new Chart(canvas, {
            type: 'line',
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
                        { position: "left", id: "Loss", scaleLabel: { display: true, labelString: "Loss" }, ticks:useMinMax?{min:0.5,max:0.68}:{} },
                        { position: "right", id: "Error", scaleLabel: { display: true, labelString: "Error"}, ticks:useMinMax?{min:0.2,max:0.5}:{} }
                    ]
                }
            }
        });
    }
}

document.addEventListener("DOMContentLoaded", retrieveData);