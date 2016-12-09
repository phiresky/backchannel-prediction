import * as mobx from "mobx";
import { observer } from "mobx-react";
import * as React from "react";
import { Styles, GUI, UIState, SingleUIState } from "./client";
import { autobind } from "core-decorators";
import * as B from "@blueprintjs/core";
import * as s from "./socket";
import * as util from "./util";
import * as Highlights from "./Highlights";
import * as Waveform from "./Waveform";
import { Feature, Utterances } from "./features";

export type VisualizerConfig = "normalizeGlobal" | "normalizeLocal" | "givenRange";

export interface VisualizerProps<T> {
    gui: GUI;
    uiState: SingleUIState;
    feature: T;
}
export abstract class Visualizer<T> extends React.Component<VisualizerProps<T>, {}> {
    preferredHeight: number;
}
export interface VisualizerConstructor<T> {
    new (props?: VisualizerProps<T>, context?: any): Visualizer<T>;
}

@observer
class CategoryTree extends React.Component<{ gui: GUI, features: s.GetFeaturesResponse, onClick: (feat: string) => void }, {}> {
    constructor(props: any) {
        super(props);
        this.currentTree = this.props.gui.categoryTree;
    }

    currentTree: B.ITreeNode[];
    @autobind @mobx.action
    handleNodeClick(n: B.ITreeNode) { if (!n.childNodes) this.props.onClick("" + n.id); }
    @autobind @mobx.action
    handleNodeExpand(n: B.ITreeNode) { n.isExpanded = true; this.forceUpdate(); }
    @autobind @mobx.action
    handleNodeCollapse(n: B.ITreeNode) { n.isExpanded = false; this.forceUpdate(); }
    render() {
        return <div>
            <B.Tree contents={this.currentTree}
                onNodeCollapse={this.handleNodeCollapse}
                onNodeClick={this.handleNodeClick}
                onNodeExpand={this.handleNodeExpand} />
            <button className="pt-button pt-popover-dismiss">Cancel</button>
        </div>;
    }
}
@observer
class LeftBar extends React.Component<{ uiState: UIState, gui: GUI }, {}> {
    static rangeOptions = ["normalizeGlobal", "normalizeLocal", "givenRange"];
    @mobx.action changeVisualizerConfig(info: SingleUIState, value: string) {
        info.config = value as VisualizerConfig;
    }
    @mobx.action changeVisualizer(info: SingleUIState, value: string) {
        info.visualizer = value as VisualizerChoice;
    }
    async changeFeature(e: React.SyntheticEvent<HTMLSelectElement>, i: number) {
        const state = this.props.gui.getDefaultSingleUIState(await this.props.gui.getFeature(e.currentTarget.value).promise);
        mobx.runInAction("changeFeature" + i, () => this.props.uiState.features[i] = state);
    }
    @mobx.action remove(uuid: number) {
        const i = this.props.uiState.features.findIndex(ui => ui.uuid === uuid);
        this.props.uiState.features.splice(i, 1);
        if (this.props.uiState.features.length === 0) {
            this.removeSelf();
        }
    }
    @autobind @mobx.action removeSelf() {
        const uis = this.props.gui.uis;
        uis.splice(uis.findIndex(ui => ui.uuid === this.props.uiState.uuid), 1);
    }
    @mobx.action async add(feat: string) {
        this.addPopover.setState({ isOpen: false });
        const gui = this.props.gui;
        const state = gui.getDefaultSingleUIState(await gui.getFeature(feat).promise);
        mobx.runInAction(() => this.props.uiState.features.push(state));
    }
    addPopover: B.Popover;
    render() {
        let minmax;
        const {uiState, gui} = this.props;
        const firstWithRange = uiState.features.find(props => props.currentRange !== null);
        if (firstWithRange && firstWithRange.currentRange) {
            minmax = [
                <div key="max" style={Styles.absoluteTopRight}>{util.round1(firstWithRange.currentRange.max)}</div>,
                <div key="min" style={Styles.absoluteBottomRight}>{util.round1(firstWithRange.currentRange.min)}</div>,
            ];
        } else minmax = "";
        const VisualizerChoices = observer((props: { info: SingleUIState }) => {
            const feature = gui.getFeature(props.info.feature).data;
            if (!feature) return <span />;
            const c = getVisualizerChoices(feature);
            if (c.length > 1) return (
                <label className="pt-label pt-inline">Visualizer
                    <div className="pt-select">
                        <select value={props.info.visualizer} onChange={e => this.changeVisualizer(props.info, e.currentTarget.value)}>
                            {c.map(c => <option key={c} value={c}>{c}</option>)}
                        </select>
                    </div>
                </label>
            );
            else return <span />;
        });
        const features = gui.getFeatures().data;
        return (
            <div className="left-bar" style={{ position: "relative", width: "100%", height: "100%" }}>
                <div style={{ ...Styles.absoluteTopLeft, paddingLeft: "5px", paddingTop: "5px" }}>
                    {uiState.features.map(info =>
                        <B.Popover key={info.uuid} interactionKind={B.PopoverInteractionKind.HOVER} popoverClassName="change-visualizer"
                            content={<div>
                                <label className="pt-label pt-inline"><button className="pt-button pt-intent-danger pt-icon-remove" onClick={e => this.remove(info.uuid)}>Remove</button></label>
                                {info.currentRange &&
                                    <label className="pt-label pt-inline">Range
                                            <div className="pt-select">
                                            <select value={info.config} onChange={e => this.changeVisualizerConfig(info, e.currentTarget.value)}>
                                                {LeftBar.rangeOptions.map(op => <option key={op} value={op}>{op}</option>)}
                                            </select>
                                        </div>
                                    </label>
                                }
                                <VisualizerChoices info={info} />
                            </div>
                            }><div className="pt-tooltip-indicator" style={{ marginBottom: "5px" }}>{info.feature}</div></B.Popover>

                    )}
                </div>
                <div style={{ ...Styles.absoluteBottomLeft, margin: "5px" }}>
                    <button onClick={this.removeSelf}>Remove</button>
                    <B.Popover content={features ? <CategoryTree gui={gui} features={features} onClick={e => this.add(e)} /> : "not loaded"}
                        interactionKind={B.PopoverInteractionKind.CLICK}
                        position={B.Position.RIGHT}
                        popoverClassName="add-visualizer"
                        useSmartPositioning={true} ref={p => this.addPopover = p}>
                        <button>Add feature</button>
                    </B.Popover>
                </div>
                {minmax}
            </div>
        );
    }
}
@observer
export class InfoVisualizer extends React.Component<{ uiState: UIState, gui: GUI }, {}> {
    render() {
        const {uiState, gui} = this.props;

        return (
            <div style={{ display: "flex", boxShadow: "-9px 10px 35px -10px rgba(0,0,0,0.29)", zIndex: 1 }}>
                <div style={Styles.leftBarCSS}>
                    <LeftBar gui={gui} uiState={uiState} />
                </div>
                <div style={{ flexGrow: 1 }}>
                    <Highlights.OverlayVisualizer gui={gui} uiState={uiState} />
                </div>
            </div>
        );

    }
}
@observer
class TextVisualizer extends Visualizer<Utterances> {
    @mobx.observable
    tooltip: number | null = null;
    @mobx.computed get playbackTooltip() {
        const data = this.props.feature.data;
        const b = util.binarySearch(0, data.length, x => +data[x].from, this.props.gui.playbackPosition * this.props.gui.totalTimeSeconds);
        return b;
    }
    // @computed currentlyVisibleTh
    getElements() {
        const width = this.props.gui.width;
        return this.props.feature.data.map((utt, i) => {
            const from = +utt.from / this.props.gui.totalTimeSeconds, to = +utt.to / this.props.gui.totalTimeSeconds;
            let left = util.getPixelFromPosition(from, 0, width, this.props.gui.zoom);
            let right = util.getPixelFromPosition(to, 0, width, this.props.gui.zoom);
            if (right < 0 || left > this.props.gui.width) return null;
            const style = {};
            if (utt.color) Object.assign(style, { backgroundColor: `rgb(${utt.color})` });
            let className = "utterance utterance-text";
            if (left < 0) {
                left = 0;
                Object.assign(style, { borderLeft: "none" });
                className += " leftcutoff";
            }
            if (right > width) {
                right = width;
                Object.assign(style, { borderRight: "none" });
                className += " rightcutoff";
            }
            const padding = 3;
            Object.assign(style, { left: left + "px", width: (right - left) + "px", padding: padding + "px" });
            return <div className={className} key={utt.id !== undefined ? utt.id : i} style={style}
                onMouseEnter={mobx.action("hoverTooltip", _ => this.tooltip = i)}
                onMouseLeave={mobx.action("hoverTooltipDisable", _ => this.tooltip = null)}>
                <span>{utt.text}</span>
            </div>;
        });
    }
    getTooltip(i: number) {
        const width = this.props.gui.width;
        const utt = this.props.feature.data[i];
        const from = +utt.from / this.props.gui.totalTimeSeconds, to = +utt.to / this.props.gui.totalTimeSeconds;
        let left = util.getPixelFromPosition(from, 0, width, this.props.gui.zoom);
        let right = util.getPixelFromPosition(to, 0, width, this.props.gui.zoom);
        if (right < 0 || left > this.props.gui.width) return null;
        let className = "utterance tooltip visible";
        let styleText;
        if (utt.color) styleText = { backgroundColor: `rgb(${utt.color})` };
        else styleText = {};
        const style = { left: left + "px", width: (right - left) + "px" };
        return <div className={className} key={utt.id} style={style}>
            <span className="content" style={styleText}><b />{utt.text}</span>
        </div>;
    }
    Tooltip = observer(function Tooltip(this: TextVisualizer) {
        return <div>
            {this.playbackTooltip !== null && this.props.gui.audioPlayer && this.props.gui.audioPlayer.playing &&
                <div style={{ position: "relative", height: "0px", width: "100%" }}>{this.getTooltip(this.playbackTooltip)}</div>}
            {this.tooltip !== null && <div style={{ position: "relative", height: "0px", width: "100%" }}>{this.getTooltip(this.tooltip)}</div>}
        </div>;
    }.bind(this));
    render() {
        return (
            <div style={{ height: "4em" }}>
                <div style={{ overflow: "hidden", position: "relative", height: "40px", width: "100%" }}>{this.getElements()}</div>
                <this.Tooltip />
            </div>
        );
    }
}

export type VisualizerChoice = "Waveform" | "Darkness" | "Text" | "Highlights";
export function getVisualizerChoices(feature: Feature): VisualizerChoice[] {
    if (!feature) return [];
    if (feature.typ === "FeatureType.SVector" || feature.typ === "FeatureType.FMatrix") {
        return ["Waveform", "Darkness"];
    } else if (feature.typ === "utterances") {
        return ["Text", "Highlights"];
    } else if (feature.typ === "highlights") {
        return ["Highlights", "Text"];
    } else throw Error("Can't visualize " + (feature as any).typ);
}

@observer
export class ChosenVisualizer extends React.Component<VisualizerProps<Feature>, {}> {
    @mobx.observable
    preferredHeight: number = 50;
    static visualizers: { [name: string]: VisualizerConstructor<any> } = {
        "Waveform": Waveform.AudioWaveform,
        "Darkness": Waveform.Darkness,
        "Text": TextVisualizer,
        "Highlights": Highlights.HighlightsVisualizer
    };
    render() {
        const Visualizer = ChosenVisualizer.visualizers[this.props.uiState.visualizer];
        return <Visualizer {...this.props} />; // ref={action("setPreferredHeight", (e: Visualizer<any>) => this.preferredHeight = e?e.preferredHeight:0)} />;
    }
}