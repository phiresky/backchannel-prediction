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
    audio: AudioContext;
    @mobx.observable
    playing: boolean;
    startedAt: number;
    duration: number;
    startPlayingAtom = new mobx.Atom("Player");
    constructor(props: any) {
        super(props);
        this.audio = new AudioContext();
        this.disposers.push(() => (this.audio as any).close());
        this.disposers.push(mobx.autorun(() => {
            // for (const feature of this.props.features) this.makeAudioBuffer(feature);
            if (this.playing) {
                for (const feature of this.props.features) {
                    const buffer = this.makeAudioBuffer(feature);
                    const audioSource = this.toAudioSource({ buffer, audio: this.audio });

                    this.duration = buffer.duration;
                    audioSource.playbackRate.value = 1;
                    const startPlaybackPosition = mobx.untracked(() => this.props.gui.playbackPosition);
                    audioSource.start(0, startPlaybackPosition * buffer.duration);
                    this.startedAt = this.audio.currentTime - startPlaybackPosition * buffer.duration;
                    audioSource.addEventListener("ended", mobx.action("audioEnded", () => this.playing = false));
                }
                requestAnimationFrame(this.updatePlaybackPosition);
            }
        }));
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
        this.props.gui.playbackPosition = (this.audio.currentTime - this.startedAt) / this.duration;
        if (this.props.gui.followPlayback) this.center();
        if (this.playing) requestAnimationFrame(this.updatePlaybackPosition);
    }
    @mobx.computed get xTranslation() {
        const x = util.getPixelFromPosition(this.props.gui.playbackPosition, this.props.gui.left, this.props.gui.width, this.props.gui.zoom);
        return "translateX(" + x + "px)";
    }
    makeAudioBuffer = mobx.createTransformer((feature: NumFeature) => {
        console.log("creating buffer for " + feature.name);
        const audioBuffer = this.audio.createBuffer(1, feature.data.shape[0], feature.samplingRate * 1000);
        feature.data.iterate("ALL", 0);
        let arr;
        if (feature.data.dataType === "float32") arr = feature.data.buffer as Float32Array;
        else arr = Float32Array.from(feature.data.buffer, v => v / 2 ** 15);
        audioBuffer.copyToChannel(arr, 0);
        return audioBuffer;
    });
    toAudioSource = mobx.createTransformer(({buffer, audio}: { buffer: AudioBuffer, audio: AudioContext }) => {
        const audioSource = audio.createBufferSource();
        audioSource.buffer = buffer;
        audioSource.playbackRate.value = 0;
        audioSource.connect(audio.destination);
        return audioSource;
    }, buf => buf.stop());

    @autobind @mobx.action
    onKeyUp(event: KeyboardEvent) {
        if (event.keyCode === 32) {
            event.preventDefault();
            this.playing = !this.playing;
        }
    }
    @autobind @mobx.action
    onKeyDown(event: KeyboardEvent) {
        if (event.keyCode === 32) {
            event.preventDefault();
        }
    }
    @autobind @mobx.action
    onClick(event: MouseEvent) {
        if (event.clientX < this.props.gui.left) return;
        event.preventDefault();
        const x = util.getPositionFromPixel(event.clientX, this.props.gui.left, this.props.gui.width, this.props.gui.zoom) !;
        this.playing = false;
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

export class AudioRecorder extends React.Component<{ gui: GUI }, {}> {
    static bufferDuration_s = 100;
    static sampleRate = 8000;
    static bufferSize = 16384;
    microphone: MediaStreamTrack;
    inputStream: MediaStream;
    source: MediaStreamAudioSourceNode;
    audioContext: AudioContext;
    processor: ScriptProcessorNode;

    constructor(props: any) {
        super(props);

    }

    private static feature: NumFeatureSVector;
    static getFeature(): NumFeatureSVector {
        return AudioRecorder.feature || (AudioRecorder.feature = {
            name: microphoneFeature.id,
            typ: "FeatureType.SVector",
            data: new Data.TwoDimensionalArray("float32", [this.bufferDuration_s * this.sampleRate, 1]),
            samplingRate: this.sampleRate / 1000,
            shift: 0,
            range: [-1, 1],
            dtype: "int16"
        });
    }
    @autobind
    async startRecording() {
        this.audioContext = new AudioContext();
        this.inputStream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: this.audioContext.sampleRate, channelCount: 1, echoCancellation: false } as any });
        const [microphone, ...others] = this.inputStream.getAudioTracks();
        if (others.length > 0) throw Error("expected single channel");
        this.microphone = microphone;
        console.log(microphone.getConstraints());
        console.log("Got microphone", microphone.label);
        this.source = this.audioContext.createMediaStreamSource(this.inputStream);
        this.processor = this.audioContext.createScriptProcessor(AudioRecorder.bufferSize, 1, 1);
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
            const promise = Resampler.nativeResample(dataSoSampled, dataSoSampled.length, this.audioContext.sampleRate, AudioRecorder.sampleRate);
            promise.then(data => {
                feat.setData(realOffset * 4, data.buffer, data.byteOffset, data.byteLength);
                realOffset += data.length;
            });
        });
        this.source.connect(this.processor);
        this.processor.connect(this.audioContext.destination);
    }
    @autobind
    stopRecording() {
        this.microphone.stop();
        this.source.disconnect(this.processor);
        this.processor.disconnect(this.audioContext.destination);
    }

    render() {
        return (
            <div><button onClick={this.startRecording}>Start Rec</button><button onClick={this.stopRecording}>Stop Rec</button></div>
        );
    }
}