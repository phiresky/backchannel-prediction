import * as React from "react";
import * as mobx from "mobx";
import { observer } from "mobx-react";
import * as util from "./util";
import { autobind } from "core-decorators";
import { NumFeature, NumFeatureSVector, FeatureID } from "./features";
import { GUI } from "./client";
import * as Data from "./Data";
import * as Resampler from "./Resampler";

@observer
export class AudioPlayer extends React.Component<{ features: NumFeatureSVector[], gui: GUI }, {}> {
    playerBar: HTMLDivElement; setPlayerBar = (p: HTMLDivElement) => this.playerBar = p;
    disposers: (() => void)[] = [];
    @mobx.observable
    playing: boolean;
    startedAt: number;
    duration: number;
    audioSources = [] as AudioBufferSourceNode[];
    constructor(props: any) {
        super(props);
    }

    center() {
        const zoom = this.props.gui.zoom;
        const w = zoom.right - zoom.left;
        let pos = this.props.gui.playbackPosition;
        if (pos - w / 2 < 0) pos = w / 2;
        if (pos + w / 2 > 1) pos = 1 - w / 2;
        zoom.left = pos - w / 2; zoom.right = pos + w / 2;
    }
    @autobind @mobx.action
    updatePlaybackPosition() {
        if (!this.playing) return;
        this.props.gui.playbackPosition = (audioContext.currentTime - this.startedAt) / this.duration;
        if (this.props.gui.followPlayback) this.center();
        if (this.playing) requestAnimationFrame(this.updatePlaybackPosition);
    }
    @mobx.computed get xTranslation() {
        const x = util.getPixelFromPosition(this.props.gui.playbackPosition, this.props.gui.left, this.props.gui.width, this.props.gui.zoom);
        return "translateX(" + x + "px)";
    }

    toAudioSource(buffer: AudioBuffer) {
        const audioSource = audioContext.createBufferSource();
        audioSource.buffer = buffer;
        audioSource.playbackRate.value = 0;
        audioSource.connect(audioContext.destination);
        return audioSource;
    }

    @mobx.action
    startPlaying() {
        this.playing = true;
        for (const feature of this.props.features) {
            const buffer = getAudioBuffer(feature);
            const audioSource = this.toAudioSource(buffer);
            this.audioSources.push(audioSource);
            this.duration = buffer.duration;
            audioSource.playbackRate.value = 1;
            const startPlaybackPosition = this.props.gui.playbackPosition;
            audioSource.start(0, startPlaybackPosition * buffer.duration);
            this.startedAt = audioContext.currentTime - startPlaybackPosition * buffer.duration;
            audioSource.addEventListener("ended", this.stopPlaying);
        }
        requestAnimationFrame(this.updatePlaybackPosition);
    }

    @autobind @mobx.action
    stopPlaying() {
        this.playing = false;
        while (this.audioSources.length > 0) this.audioSources.pop() !.stop();
    }
    @autobind
    onKeyUp(event: KeyboardEvent) {
        if (event.keyCode === 32) {
            event.preventDefault();
            if (this.playing) this.stopPlaying();
            else this.startPlaying();
        }
    }
    @autobind
    onKeyDown(event: KeyboardEvent) {
        if (event.keyCode === 32) {
            event.preventDefault();
        }
    }
    @autobind
    onClick(event: MouseEvent) {
        if (event.clientX < this.props.gui.left) return;
        event.preventDefault();
        const x = util.getPositionFromPixel(event.clientX, this.props.gui.left, this.props.gui.width, this.props.gui.zoom) !;
        if (this.playing) this.stopPlaying();
        mobx.runInAction("clickSetPlaybackPosition", () => this.props.gui.playbackPosition = Math.max(x, 0));
    }

    componentDidMount() {
        const uisDiv = this.props.gui.uisDiv;
        uisDiv.addEventListener("click", this.onClick);
        window.addEventListener("keydown", this.onKeyDown);
        window.addEventListener("keyup", this.onKeyUp);
        this.disposers.push(() => uisDiv.removeEventListener("click", this.onClick));
        this.disposers.push(() => window.removeEventListener("keydown", this.onKeyDown));
        this.disposers.push(() => window.removeEventListener("keyup", this.onKeyUp));
        this.disposers.push(mobx.action("stopPlaybackExit", () => this.playing = false));
    }
    componentWillUnmount() {
        for (const disposer of this.disposers) disposer();
    }
    render() {
        return (
            <div ref={this.setPlayerBar} style={{ position: "fixed", width: "2px", height: "100vh", top: 0, left: 0, transform: this.xTranslation, backgroundColor: "gray" }} />
        );
    }
}

@observer
export class PlaybackPosition extends React.Component<{ gui: GUI }, {}> {
    render() {
        const gui = this.props.gui;
        return <span>{(gui.playbackPosition * gui.totalTimeSeconds).toFixed(4)}</span>;
    }
}

export const microphoneFeature = {
    id: "/microphone" as any as FeatureID,
    name: "microphone"
};

const cache = new WeakMap<NumFeatureSVector, AudioBuffer>();

export function fillAudioBuffer(source: Float32Array | Int16Array, target: Float32Array) {
    if (source instanceof Float32Array) target.set(source);
    else {
        for (let i = 0; i < source.length; i++) {
            target[i] = source[i] / 2 ** 15;
        }
    }
}
export function getAudioBuffer(feat: NumFeatureSVector) {
    const buf = cache.get(feat);
    if (buf) {
        console.log("got cached AudioBuffer");
        return buf;
    } else {
        console.log("creating buffer for " + feat.name);
        const buffer = audioContext.createBuffer(1, feat.data.shape[0], feat.samplingRate * 1000);
        const audioBufferData = buffer.getChannelData(0);
        fillAudioBuffer(feat.data.buffer, audioBufferData);
        if (feat.data.dataType === "float32") {
            // data types are compatible, make the buffer shared so writes directly affect the audio buffer
            feat.data.buffer = audioBufferData;
        }
        cache.set(feat, buffer);
        mobx.reaction(feat.name + " data changed", () => feat.data.iterate("ALL", 0), () => {
            // only update if it isn't a shared buffer
            if (feat.data.buffer !== audioBufferData) {
                console.log("update audio buffer for " + feat.name);
                fillAudioBuffer(feat.data.buffer, audioBufferData);
            }
        });
        return buffer;
    }
}
const audioContext = new AudioContext();

export class AudioRecorder extends React.Component<{ gui: GUI }, {}> {
    static bufferDuration_s = 100;
    static sampleRate = 8000;
    static bufferSize = 16384;
    microphone: MediaStreamTrack;
    inputStream: MediaStream;
    source: MediaStreamAudioSourceNode;
    processor: ScriptProcessorNode;


    constructor(props: any) {
        super(props);
    }

    private static feature: NumFeatureSVector;
    static getFeature(): NumFeatureSVector {
        const count = this.bufferDuration_s * this.sampleRate;
        const buffer = audioContext.createBuffer(1, count, this.sampleRate);
        const data = new Data.TwoDimensionalArray("float32", [count, 1], buffer.getChannelData(0));
        (data.buffer as Float32Array).fill(NaN);
        return AudioRecorder.feature || (AudioRecorder.feature = {
            name: microphoneFeature.id,
            typ: "FeatureType.SVector",
            data,
            samplingRate: this.sampleRate / 1000,
            shift: 0,
            range: [-1, 1],
            dtype: "int16"
        });
    }
    @autobind
    async startRecording() {
        this.inputStream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: audioContext.sampleRate, channelCount: 1, echoCancellation: false } as any });
        const [microphone, ...others] = this.inputStream.getAudioTracks();
        if (others.length > 0) throw Error("expected single channel");
        this.microphone = microphone;
        console.log(microphone.getConstraints());
        console.log("Got microphone", microphone.label);
        this.source = audioContext.createMediaStreamSource(this.inputStream);
        this.processor = audioContext.createScriptProcessor(AudioRecorder.bufferSize, 1, 1);
        const feat = AudioRecorder.getFeature().data;
        let lastOffset = 0;
        let realOffset = 0;
        let lastTime = 0;
        this.processor.addEventListener("audioprocess", event => {
            const dataSoSampled = event.inputBuffer.getChannelData(0);
            //const offset = (((event.playbackTime % AudioRecorder.bufferDuration_s) * AudioRecorder.sampleRate) | 0);
            //console.log(event.playbackTime, offset, offset - lastOffset, data.length, Math.round(data.length/(event.playbackTime-lastTime))|0, data);
            //lastOffset = offset;
            //lastTime = event.playbackTime;
            const promise = Resampler.nativeResample(dataSoSampled, dataSoSampled.length, audioContext.sampleRate, AudioRecorder.sampleRate);
            promise.then(data => {
                feat.setData(realOffset * 4, data.buffer, data.byteOffset, data.byteLength);
                realOffset += data.length;
            });
        });
        this.source.connect(this.processor);
        this.processor.connect(audioContext.destination);
    }
    @autobind
    stopRecording() {
        this.microphone.stop();
        this.source.disconnect(this.processor);
        this.processor.disconnect(audioContext.destination);
    }

    render() {
        return (
            <div><button onClick={this.startRecording}>Start Rec</button><button onClick={this.stopRecording}>Stop Rec</button></div>
        );
    }
}