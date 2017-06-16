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

    center(offset = 0.5) {
        const zoom = this.props.gui.zoom;
        const w = zoom.right - zoom.left;
        let pos = this.props.gui.playbackPosition;
        if (pos - w * offset < 0) pos = w * offset;
        if (pos + w * (1 - offset) > 1) pos = 1 - w * (1 - offset);
        zoom.left = pos - w * offset; zoom.right = pos + w * (1 - offset);
    }
    @autobind @mobx.action
    updatePlaybackPosition() {
        if (!this.playing) return;
        const newPos = ((audioContext.currentTime - this.startedAt) / this.duration) % 1;
        if (newPos > this.props.gui.selectionEnd) {
            this.props.gui.playbackPosition = this.props.gui.selectionEnd;
            this.stopPlaying();
            return;
        }
        this.props.gui.playbackPosition = newPos;
        if (this.props.gui.followPlayback) this.center(this.props.gui.followCenter);
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
        if (this.props.gui.playbackPosition === this.props.gui.selectionEnd)
            this.props.gui.playbackPosition = this.props.gui.selectionStart;
        for (const feature of this.props.features) {
            const buffer = getAudioBuffer(feature);
            const audioSource = this.toAudioSource(buffer);
            this.audioSources.push(audioSource);
            this.duration = buffer.duration;
            audioSource.playbackRate.value = 1;
            const startPlaybackPosition = this.props.gui.playbackPosition;
            audioSource.loop = true;
            audioSource.start(0, startPlaybackPosition * buffer.duration);
            this.startedAt = audioContext.currentTime - startPlaybackPosition * buffer.duration;
            audioSource.addEventListener("ended", this.stopPlaying);
        }
        requestAnimationFrame(this.updatePlaybackPosition);
    }

    @autobind @mobx.action
    stopPlaying() {
        this.playing = false;
        while (this.audioSources.length > 0) this.audioSources.pop()!.stop();
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
        if (this.playing) this.stopPlaying();
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
            <div ref={this.setPlayerBar} style={{ position: "fixed", width: "2px", height: "100vh", top: 0, left: 0, transform: this.xTranslation, backgroundColor: "black" }} />
        );
    }
}

@observer
export class RegionSelector extends React.Component<{ gui: GUI }, {}> {
    div: HTMLDivElement; setDiv = (p: HTMLDivElement) => this.div = p;
    disposers: (() => void)[] = [];

    @autobind @mobx.action
    onMouseDown(event: MouseEvent) {
        if (event.clientX < this.props.gui.left) return;
        const x = util.getPositionFromPixel(event.clientX, this.props.gui.left, this.props.gui.width, this.props.gui.zoom)!;
        this.props.gui.selectionStart = x;
        this.props.gui.selectionEnd = x;
        window.addEventListener("mousemove", this.onMouseMove);
    }

    @autobind @mobx.action
    onMouseMove(event: MouseEvent) {
        const gui = this.props.gui;
        event.preventDefault();
        const x = util.getPositionFromPixel(event.clientX, gui.left, gui.width, gui.zoom)!;
        if (x < gui.selectionStart) {
            gui.selectionStart = x;
        } else {
            gui.selectionEnd = x;
        }
    }

    @autobind @mobx.action
    onMouseUp(event: MouseEvent) {
        window.removeEventListener("mousemove", this.onMouseMove);
        const gui = this.props.gui;
        if (isNaN(gui.selectionStart)) return;
        gui.playbackPosition = gui.selectionStart;
        const diff = util.getPixelFromPosition(Math.abs(gui.selectionStart - gui.selectionEnd), 0, this.props.gui.width, { left: 0, right: gui.zoom.right - gui.zoom.left });
        if (diff < 5) {
            gui.selectionStart = NaN;
            gui.selectionEnd = NaN;
        }
    }

    componentDidMount() {
        const uisDiv = this.props.gui.uisDiv;
        window.addEventListener("mousedown", this.onMouseDown);
        window.addEventListener("mouseup", this.onMouseUp);
        this.disposers.push(() => uisDiv.removeEventListener("mousedown", this.onMouseDown));
        this.disposers.push(() => window.removeEventListener("mouseup", this.onMouseUp));
    }
    componentWillUnmount() {
        for (const disposer of this.disposers) disposer();
    }

    render() {
        let a = this.props.gui.selectionStart;
        let b = this.props.gui.selectionEnd;
        if (isNaN(a) || isNaN(b)) return <div />;
        const left = util.getPixelFromPosition(a, this.props.gui.left, this.props.gui.width, this.props.gui.zoom);
        const right = util.getPixelFromPosition(b, this.props.gui.left, this.props.gui.width, this.props.gui.zoom);
        return (
            <div ref={this.setDiv} style={{ position: "fixed", height: "100vh", top: 0, left, width: right - left, backgroundColor: "rgba(0,0,0,0.2)" }} />
        );
    }
}

@observer
export class PlaybackPosition extends React.Component<{ gui: GUI }, {}> {
    render() {
        const fix = (x: number) => (x * gui.totalTimeSeconds).toFixed(4);
        const gui = this.props.gui;
        if (!isNaN(gui.selectionStart) && !isNaN(gui.selectionEnd) && Math.abs(gui.selectionEnd - gui.selectionStart) > 1e-6)
            return <span>[{fix(gui.selectionStart)}s — {fix(gui.playbackPosition)}s — {fix(gui.selectionEnd)}]
                Duration: {fix(gui.selectionEnd - gui.selectionStart)}s</span>;
        else
            return <span>{(gui.playbackPosition * gui.totalTimeSeconds).toFixed(4)}</span>;
    }
}

export const microphoneFeature = {
    id: "/microphone/extracted/adc" as any as FeatureID,
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

@observer
export class AudioRecorder extends React.Component<{ gui: GUI }, {}> {
    static bufferDuration_s = 60; // must be same as server side (MicrophoneHandler.buffer_duration_s)
    // native resample has a bug in chrome where it crashes after ~30s (https://bugs.chromium.org/p/chromium/issues/detail?id=429806)
    // multitap
    static resampler = Resampler.multiTapInterpolate;
    static doResample = true;
    static sampleRate = AudioRecorder.doResample ? 8000 : audioContext.sampleRate;
    @mobx.observable processingBufferDuration_s = "0.15";
    microphone: MediaStreamTrack;
    inputStream: MediaStream;
    source: MediaStreamAudioSourceNode;
    processor: ScriptProcessorNode;
    @mobx.observable recording = false;
    recordingStartTime = NaN;
    @mobx.observable playthrough = false;
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
    gotData(feat: Data.TwoDimensionalArray, event: AudioProcessingEvent, resampledData: Float32Array) {
        let shouldBeOffset = Math.round((event.playbackTime - this.recordingStartTime) * AudioRecorder.sampleRate);
        const endOffset = shouldBeOffset + resampledData.length;
        if (endOffset > feat.shape[0]) {
            // do wrap around
            shouldBeOffset = 0;
            this.recordingStartTime = event.playbackTime;
            this.props.gui.getFeature("/microphone/extracted/nn.smooth.bc").then(x => (x.data as Data.TwoDimensionalArray).fill(0));
        }
        feat.setData(shouldBeOffset, resampledData);
        this.props.gui.socketManager.sendFeatureSegment({
            conversation: null as any,
            feature: microphoneFeature.id,
            byteOffset: shouldBeOffset * Float32Array.BYTES_PER_ELEMENT
        }, resampledData);
    }
    @autobind
    async startRecording() {
        const bufferSize = 2 ** Math.round(Math.log2(audioContext.sampleRate * +this.processingBufferDuration_s));

        this.inputStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                mandatory: {
                    googEchoCancellation: false,
                    googAutoGainControl: false,
                    googNoiseSuppression: false,
                    googHighpassFilter: true,
                    googTypingNoiseDetection: false,
                    echoCancellation: false
                },
            } as any
        });
        const [microphone, ...others] = this.inputStream.getAudioTracks();
        if (others.length > 0) throw Error("expected single channel");
        this.microphone = microphone;
        console.log(microphone.getConstraints());
        console.log("Got microphone", microphone.label);
        this.source = audioContext.createMediaStreamSource(this.inputStream);
        this.processor = audioContext.createScriptProcessor(bufferSize, 1, 1);
        const feat = AudioRecorder.getFeature().data;
        this.recordingStartTime = audioContext.currentTime;
        this.processor.addEventListener("audioprocess", event => {
            const dataSoSampled = event.inputBuffer.getChannelData(0);
            if (AudioRecorder.doResample) {
                const data = AudioRecorder.resampler(dataSoSampled, dataSoSampled.length, audioContext.sampleRate, AudioRecorder.sampleRate);
                this.gotData(feat, event, data);
            } else this.gotData(feat, event, dataSoSampled);
        });
        this.source.connect(this.processor);
        this.processor.connect(audioContext.destination);
        this.props.gui.getFeature("/microphone/extracted/nn.smooth.bc").then(x => (x.data as Data.TwoDimensionalArray).fill(0));
        mobx.runInAction("startRecording", () => {
            this.recording = true;
            this.props.gui.playbackPosition = 0;
            setTimeout(() => this.props.gui.audioPlayer && this.props.gui.audioPlayer.startPlaying(), 5);
        });
    }
    @autobind @mobx.action
    stopRecording() {
        this.recording = false;
        this.microphone.stop();
        this.source.disconnect(this.processor);
        this.processor.disconnect(audioContext.destination);
        this.props.gui.audioPlayer && this.props.gui.audioPlayer.stopPlaying();
    }
    @mobx.action
    togglePlaythrough = (e: React.SyntheticEvent<HTMLInputElement>) => {
        this.props.gui.playthrough = e.currentTarget.checked;
    }
    @mobx.action
    setBufferDuration = (e: React.SyntheticEvent<HTMLInputElement>) => {
        this.processingBufferDuration_s = e.currentTarget.value;
    }

    render() {
        return (
            <div>{this.recording
                ? <button onClick={this.stopRecording}>Stop Rec</button>
                : <button onClick={this.startRecording}>Start Rec</button>
            }
                <p><label><input type="checkbox" checked={this.props.gui.playthrough} onChange={this.togglePlaythrough} /> Playthrough while recording</label></p>
                <p><label>Record buffer size: <input type="number" value={this.processingBufferDuration_s} onChange={this.setBufferDuration} step={0.02} />s</label></p>
            </div>
        );
    }
}