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
    @observable doGameStudy = false; // :(
    @observable samples: common.BCSamples[] = [];
    @observable chosenSample: string;
    @observable chosenSamplesAudio: HTMLAudioElement[] = [];
    @observable monosegs: string[] = [];
    @observable netRatingSegments: string[] = [];
    @observable netRatings: Map<string, number> = observable.map() as any;
    @observable currentMonoseg: number = -1;
    @observable state: "loading" | "beforeGame" | "ingame" | "after" | "rateNet";
    @observable hasBCedOnce = false;
    @observable sessionId: number;
    @observable bcCount = 0;
    @observable bcsLoaded = false;
    @observable nextSegment: HTMLAudioElement | null;
    preload(files: string[]) {
        const total = files.length;
        let done = 0;
        return files.map(path => {
            const audio = new Audio(path);
            audio.addEventListener("canplaythrough", e => {
                done++;
                if (done === total) this.bcsLoaded = true;
            });
            return audio;
        });
    }
    constructor(public socket: common.RouletteClientSocket) {
        this.state = "loading";

        socket.emit("getData", {}, ({ bcSamples, monosegs, netRatingSegments, sessionId }) => {
            this.samples = bcSamples;
            this.monosegs = monosegs;
            this.netRatingSegments = netRatingSegments;
            this.sessionId = sessionId;
            console.log("segments: ", netRatingSegments);
            if (this.doGameStudy) this.chosenSample = _.sample(mobx.toJS(bcSamples)).name;
            this.state = (this.doGameStudy && Math.random() < 0.5) ? "beforeGame" : "rateNet";
        });
        mobx.autorun(() => {
            if (!this.chosenSample) return;
            const found = this.samples.find(e => e.name === this.chosenSample);
            if (!found) return;
            this.bcsLoaded = false;
            this.chosenSamplesAudio = this.preload(found.samples);
        });
        mobx.autorun(() => {
            if (!this.doGameStudy) return;
            const i = this.currentMonoseg + 1;
            // for preloading
            if (0 <= i && i < this.monosegs.length)
                this.nextSegment = new Audio(this.monosegs[i]);
        });
    }
    playBackchannel() {
        if (!(this.state === "beforeGame" || this.currentMonoseg >= 0)) return;
        if (this.chosenSamplesAudio.length === 0) console.error("no samples, cant play");

        const audio = _.sample(mobx.toJS(this.chosenSamplesAudio));
        audio.pause();
        audio.currentTime = 0;
        audio.play();
        this.hasBCedOnce = true;
        this.bcCount++;
        if (this.currentMonoseg >= 0 && this.currentAudio) {
            const time = this.currentAudio.currentTime;
            if (time < this.currentAudio.duration)
                this.socket.emit("submitBC", { segment: this.monosegs[this.currentMonoseg], time, duration: this.currentAudio.duration }, () => { });
        }
    }
    submitNetRatings() {
        const entries = Array.from(this.netRatings.entries());
        this.socket.emit("submitNetRatings", { segments: entries, final: true }, () => {
            this.netRatingsSubmitted = true;
            if (!this.hasBCedOnce && this.doGameStudy) this.state = "beforeGame";
            else this.state = "after";
        });
    }
    beginGame() {
        this.socket.emit("beginStudy", { bcSampleSource: this.chosenSample }, data => {
            this.currentMonoseg++;
            this.state = "ingame";
        });
    }
    @observable x = 10;
    @observable y = NaN;
    @observable netRatingsSubmitted = false;
    @observable currentAudio: HTMLAudioElement | null;
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
@mobxReact.observer
class Segment extends React.Component<{ store: Store, segment: string }, {}> {
    random = Math.random();
    setRating = (segment: string, rating: string) => {
        this.props.store.netRatings.set(segment, +rating);
        this.props.store.socket.emit("submitNetRatings", { segments: [[segment, +rating]], final: false }, () => { });
    }
    render() {
        const { store, segment } = this.props;
        return (
            <div>
                <div><audio src={segment + "?" + this.random} controls style={{ width: "100%", marginBottom: "1em" }} /></div>
                Very Unnatural
                <B.RadioGroup
                    //label="Your rating"
                    className="pt-control pt-inline"
                    //className="pt-inline"
                    onChange={(e: React.SyntheticEvent<HTMLInputElement>) => this.setRating(segment, e.currentTarget.value)}
                    selectedValue={store.netRatings.get(segment) + ""}
                >
                    {...[1, 2, 3, 4, 5].map(r => <B.Radio className="pt-inline" key={r} label={"" + r} value={"" + r} />)}
                </B.RadioGroup>
                <span style={{ marginLeft: "-20px" }}>Completely Natural</span>
                <hr />
            </div>
        );
    }
}
class NetRatingScreen extends Component {
    submit = () => this.props.store.submitNetRatings();
    @mobx.computed get canSubmit() {
        return this.props.store.netRatings.size >= 5;
    }
    render() {
        const store = this.props.store;
        return (
            <div>
                <p>Listen to the following conversations. One person is talking about a topic, another person is listening and giving backchannel feedback (e.g. "uh-hum", "yeah", "right").</p>
                <p>Rate the way the <i>listener</i> sounds from 1 ("very unnatural") to 5 ("completely natural").</p>
                <hr />
                {store.netRatingSegments.map(segment => <Segment key={segment} segment={segment} store={store} />)}
                {!this.canSubmit && <div className="pt-callout pt-intent-warning">You need to rate at least 5 of the above to be able to submit</div>}
                <button className="pt-button pt-intent-primary" disabled={!this.canSubmit} onClick={this.submit}>Submit</button>
            </div>
        )
    }
}
class InGame extends Component {
    @observable curTime = NaN;
    @observable totalTime = NaN;
    interval: any;
    @observable gameState: "loading" | "inprogress" | "finished" = "loading";

    nextAudio = () => {
        const store = this.props.store;
        this.gameState = "loading";
        if (store.currentMonoseg >= store.monosegs.length) {
            if (store.netRatingsSubmitted) {
                store.state = "after";
            } else {
                store.state = "rateNet";
            }
            return;
        }
        store.currentAudio!.src = store.monosegs[store.currentMonoseg];
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
        store.currentMonoseg++;
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
    get cur() {
        if (this.gameState === "finished") return this.props.store.currentMonoseg - 1;
        else return this.props.store.currentMonoseg;
    }
    render() {
        const store = this.props.store;
        return (
            <div>
                <div>
                    Sample: {this.cur} of {store.monosegs.length}
                    <B.ProgressBar value={this.cur / store.monosegs.length} intent={B.Intent.PRIMARY}
                        className="pt-no-animation"
                    />
                </div>
                <SpacebarTrigger store={store} />
                <div>Playback: {this.curTime.toFixed(0)}&thinsp;s / {this.totalTime.toFixed(0)}&thinsp;s
                <B.ProgressBar value={this.curTime ? this.curTime / this.totalTime : 0} intent={B.Intent.SUCCESS}
                        className="pt-no-animation"
                    />
                </div>
                {/*<button onClick={() => store.playBackchannel()}>BC</button>*/}
                <audio ref={a => store.currentAudio = a} onEnded={this.audioDone} onCanPlayThrough={this.loadingComplete} />
                <mobxReact.Observer>{() => {
                    switch (this.gameState) {
                        case "loading":
                            return <span>Loading audio</span>/*<button onClick={this.play}>Play audio</button>*/
                        case "inprogress":
                            return <div><p>Press space to say uh-huh whenever you feel like it</p><p>BC count: {store.bcCount}</p></div>
                        case "finished":
                            return <div>Done! <B.Button onClick={this.nextAudio}>Go to next sample</B.Button></div>
                    }
                }}
                </mobxReact.Observer>
                <div style={{ marginTop: "2em" }}><small>Session: {store.sessionId} | Sample: {store.monosegs[this.cur]}</small></div>
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
class BeforeGame extends Component {
    componentDidMount() {
        const store = this.props.store;
        if (!store.chosenSample)
            store.chosenSample = _.sample(mobx.toJS(store.samples)).name;
    }
    render() {
        const store = this.props.store;
        return (
            <div>
                <SpacebarTrigger store={store} />
                <p>You will hear some ~30 seconds long segments of speech.</p><p>Listen to them by pressing the space bar to say an acknowledgment such as "uh-huh", "yeah", "right".</p>
                <Select value={ref(store, "chosenSample")} options={store.samples.map(sample => sample.name)} label="Choose your voice:" />
                <span>Press the space bar to try it out. Make sure you can hear something, then press begin.</span>
                <div><B.Button disabled={!store.hasBCedOnce || !store.bcsLoaded} onClick={() => store.beginGame()}>{store.bcsLoaded ? "Begin" : "Loading..."}</B.Button></div>
            </div>
        );
    }
}
class After extends Component {
    @observable text = "";
    @observable textSubmitted = false;
    submit = () => {
        this.props.store.socket.emit("comment", this.text, () => {
            this.textSubmitted = true;
        });
    }
    render() {
        const store = this.props.store;
        return (
            <div>
                <p>Thank you for participating!</p>
                <p>Your session id: {store.sessionId}</p>
                {!this.textSubmitted ?
                    <div><p>If you encountered any problems or have some other comment:</p>
                        <textarea className="pt-input pt-fill" value={this.text} onChange={e => this.text = e.currentTarget.value} />
                        <p><button className="pt-button pt-intent-primary" onClick={this.submit}>Submit</button></p></div>
                    : ""}

            </div>
        )
    }
}
class GUI extends Component {
    inner() {
        const store = this.props.store;
        switch (store.state) {
            case 'loading': return <div>Loading...</div>;
            case 'beforeGame': return <BeforeGame store={store} />;
            case 'rateNet': return <NetRatingScreen store={store} />;
            case 'ingame': return <InGame store={store} />;
            case 'after': return <After store={store} />;
        }
    }
    render() {
        return (
            <div style={{ maxWidth: "800px", margin: "0 auto", marginTop: "1em" }}>
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
    const _socket = io();
    const socket = _socket as any as common.RouletteClientSocket;
    store = new Store(socket);
    gui = render(<GUI store={store} />, document.getElementById("app"));
    Object.assign(window, { store, gui, mobx, mobxReact });
});
