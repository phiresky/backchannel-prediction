import { Component as _Component } from 'react';
import * as React from 'react';
import { render } from 'react-dom';
import * as mobx from 'mobx';
import { observable } from 'mobx';
import * as mobxReact from 'mobx-react';
import * as io from 'socket.io-client';
import * as common from './common';
import * as _ from 'lodash';
import * as B from '@blueprintjs/core';
import "@blueprintjs/core/dist/blueprint.css";

class Store {
    @observable samples: common.BCSamples[] = [];
    @observable chosenSample: string;
    @observable chosenSamplesAudio: HTMLAudioElement[] = [];
    @observable monosegs: string[] = [];
    @observable doneMonosegs: string[] = [];
    @observable state: "intro" | "ingame" = "intro";
    @observable hasBCedOnce = false;
    @observable sessionId: number;
    @observable bcCount = 0;
    constructor(public socket: common.RouletteClientSocket) {
        socket.emit("getBCSamples", {}, samples => {
            this.samples = samples;
            this.chosenSample = _.sample(mobx.toJS(samples)).name;
            console.log(this.chosenSample);
        });
        socket.emit("getMonosegs", {}, monosegs => this.monosegs = monosegs);
        mobx.autorun(() => {
            if (!this.chosenSample) return;
            const found = this.samples.find(e => e.name === this.chosenSample);
            if (!found) return;
            this.chosenSamplesAudio = found.samples.map(path => new Audio(path));
        });
    }
    playBackchannel() {
        if (!(this.state === "intro" || this.currentSegment)) return;
        if (this.chosenSamplesAudio.length === 0) console.error("no samples, cant play");

        const audio = _.sample(mobx.toJS(this.chosenSamplesAudio));
        audio.pause();
        audio.currentTime = 0;
        audio.play();
        this.hasBCedOnce = true;
        this.bcCount++;
        if (this.currentSegment && this.currentAudio) {
            const t = this.currentAudio.currentTime;
            this.socket.emit("submitBC", { segment: this.currentSegment, time: t }, () => {

            });
        }
    }
    begin() {
        this.socket.emit("beginStudy", { bcSampleSource: this.chosenSample }, data => {
            this.state = "ingame";
            this.sessionId = data.sessionId;
        });
    }
    @observable x = 10;
    @observable y = NaN;
    @observable currentAudio: HTMLAudioElement | null;
    @observable currentSegment: string | null;
}

@mobxReact.observer
class Component extends React.Component<{ store: Store }, {}> {

}
type Reference<T> = { get: () => T, set: (v: T) => void };

function ref<T, K extends keyof T>(store: T, key: K): Reference<T[K]> {
    return {
        get: () => store[key],
        set: (val: T[K]) => store[key] = val
    }
}

const Input = mobxReact.observer(function Input({ value }: { value: Reference<string> }) {
    return <input value={value.get()} onChange={e => value.set(e.currentTarget.value)} />;
});

const Select = mobxReact.observer(function Select({ value, options, label }: { value: Reference<string>, options: string[], label: string }) {
    return (
        <label className="pt-label pt-inline">
            {label}
            <div className="pt-select">
                <select value={value.get()} onChange={e => value.set(e.currentTarget.value)}>
                    {options.map(option => <option key={option} value={option}>{option}</option>)}
                </select>
            </div>
        </label>
    );
});


class InGame extends Component {
    @observable curTime = NaN;
    @observable totalTime = NaN;
    interval: any;
    @observable gameState: "loading" | "inprogress" | "finished" = "loading";

    nextAudio = () => {
        const store = this.props.store;
        this.gameState = "loading";
        store.currentSegment = _.sample(mobx.toJS(this.props.store.monosegs));
        store.currentAudio!.src = store.currentSegment;
        store.bcCount = 0;
    }
    play = () => {
        const store = this.props.store;
        store.currentAudio!.play();
        this.gameState = "inprogress";
    }
    audioDone = () => {
        const store = this.props.store;
        this.gameState = "finished";
        this.props.store.doneMonosegs.push(store.currentSegment!);
    }
    loadingComplete = () => {
        this.play();
    }
    componentDidMount() {
        this.interval = setInterval(() => this.setTimer(), 100);
        this.nextAudio();
    }
    componentWillUnmount() {
        clearInterval(this.interval);
    }

    setTimer() {
        const store = this.props.store;
        if (store.currentAudio) {
            this.curTime = store.currentAudio.currentTime;
            this.totalTime = store.currentAudio.duration;
        } else {
            this.curTime = NaN;
            this.totalTime = NaN;
        }
    }
    render() {
        const store = this.props.store;
        return (
            <div>
                <div>
                    Sample: {store.doneMonosegs.length} of {store.monosegs.length}
                <B.ProgressBar value={store.doneMonosegs.length / store.monosegs.length} intent={B.Intent.PRIMARY}
                    className="pt-no-animation"
                />
                </div>
                <SpacebarTrigger store={store} />
                <div>Playback: {this.curTime.toFixed(0)}&thinsp;s / {this.totalTime.toFixed(0)}&thinsp;s
                <B.ProgressBar value={this.curTime ? this.curTime/this.totalTime: 0} intent={B.Intent.SUCCESS}
                    className="pt-no-animation"
                />
                </div>
                {/*<button onClick={() => store.playBackchannel()}>BC</button>*/}
                <audio ref={a => store.currentAudio = a} onEnded={this.audioDone} onCanPlayThrough={this.loadingComplete} />
                {(() => {
                    switch (this.gameState) {
                        case "loading":
                            return <span>Loading audio</span>/*<button onClick={this.play}>Play audio</button>*/
                        case "inprogress":
                            return <span>Press space to say uh-huh whenever you feel like it</span>
                        case "finished":
                            return <div>Done! <B.Button onClick={this.nextAudio}>Go to next sample</B.Button></div>
                    }
                })()}
                <div style={{ marginTop: "2em" }}><small>Session: {store.sessionId} | Sample: {store.currentSegment}</small></div>
            </div>
        );
    }
}

class SpacebarTrigger extends Component {
    keydown = (e: KeyboardEvent) => {
        if (e.which === 32 || e.keyCode === 32) {
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
            this.props.store.playBackchannel();
        }
    }
    ignoreSpaces = (e: KeyboardEvent) => {
        if (e.which === 32 || e.keyCode === 32) {
            e.stopImmediatePropagation();
            e.preventDefault();
            e.stopImmediatePropagation();
        }
    }
    componentDidMount() {
        document.addEventListener("keydown", this.keydown, true);
        document.addEventListener("keypress", this.ignoreSpaces, true);
        document.addEventListener("keyup", this.ignoreSpaces, true);
    }
    componentWillUnmount() {
        document.removeEventListener("keydown", this.keydown, true);
        document.removeEventListener("keypress", this.ignoreSpaces, true);
        document.removeEventListener("keyup", this.ignoreSpaces, true);

    }
    render() {
        return <span />;
    }
}
class Welcome extends Component {
    render() {
        const store = this.props.store;
        return (
            <div>
                <SpacebarTrigger store={store} />
                <p>You will listen to some segments of speech.</p><p>Pretend you are listening by pressing the space bar to say "uh-huh".</p>
                <Select value={ref(store, "chosenSample")} options={store.samples.map(sample => sample.name)} label="Choose your voice:"/>
                <span>Press the space bar to try it out</span>
                <div><B.Button disabled={!store.hasBCedOnce} onClick={() => store.begin()}>Begin</B.Button></div>
            </div>
        );
    }
}
class GUI extends Component {
    inner() {
        const store = this.props.store;
        switch (store.state) {
            case 'intro': return <Welcome store={store} />;
            case 'ingame': return <InGame store={store} />;
        }
    }
    render() {
        return (
            <div style={{ maxWidth: "900px", margin: "0 auto" }}>
                <h1>Backchannel Survey</h1>
                <hr />
                {this.inner()}
            </div>
        );
    }
}

let gui;
let store;
document.addEventListener("DOMContentLoaded", () => {
    const div = document.createElement("div");
    const _socket = io();
    const socket = _socket as any as common.RouletteClientSocket;
    store = new Store(socket);
    document.body.appendChild(div);
    gui = render(<GUI store={store} />, div);
    Object.assign(window, { store, gui, mobx, mobxReact });
});
