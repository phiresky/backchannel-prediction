import { observer } from "mobx-react";
import * as mobx from "mobx";
import * as React from "react";

import * as c from "./client";
import * as v from "./Visualizer";
import * as util from "./util";
import { autobind } from "core-decorators";
import { Highlights } from "./features";

@observer
export class OverlayVisualizer extends React.Component<{ gui: c.GUI, uiState: c.UIState }, {}> {
    @mobx.observable
    container: HTMLDivElement;
    setContainer = (e: HTMLDivElement) => this.container = e;
    @mobx.observable
    visses: number[] = [];
    @mobx.computed get preferredHeight() {
        const maxHeight = Math.max(...this.visses.filter(x => x !== null));
        if (isFinite(maxHeight)) return maxHeight;
        else return c.globalConfig.emptyVisHeight;
    }
    @mobx.computed get height() {
        return this.props.uiState.height === "auto" ? this.preferredHeight : this.props.uiState.height;
    }
    @autobind
    mouseDown(e: React.MouseEvent<HTMLDivElement>) {
        e.preventDefault();
        const startY = e.clientY;
        const heightBefore = this.height;
        const moveListener = mobx.action("dragresize", (e2: MouseEvent) => this.props.uiState.height = heightBefore + e2.clientY - startY);
        window.addEventListener("mousemove", moveListener);
        const removeListener = (e: Event) => {
            e.preventDefault();
            window.removeEventListener("mousemove", moveListener);
            window.removeEventListener("mouseup", removeListener);
        };
        window.addEventListener("mouseup", removeListener);
    }
    render() {
        const ui = this.props.uiState;
        const gui = this.props.gui;
        const self = this;
        const SingleOverlay = observer(function SingleOverlay({state, i}: { state: c.SingleUIState, i: number }) {
            const feature = gui.getFeature(state.feature).data;
            return (
                <div style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%" }}>
                    {feature && <v.ChosenVisualizer feature={feature} gui={gui} uiState={state} ref={mobx.action("setChosenVis", (d: v.ChosenVisualizer) => self.visses[i] = d && d.preferredHeight)} />}
                </div>
            );
        });
        return (
            <div style={{ position: "relative", height: this.height, minHeight: "20px", width: "100%" }}>
                {ui.features.map((state, i) => <SingleOverlay key={state.uuid} state={state} i={i} />)}
                <div onMouseDown={this.mouseDown} style={{ position: "absolute", height: "10px", bottom: "-5px", width: "100%", cursor: "ns-resize" }}></div>
            </div>
        );
    }
}
@observer
export class HighlightsVisualizer extends React.Component<v.VisualizerProps<Highlights>, {}> {
    preferredHeight = 0;
    getElements() {
        const width = this.props.gui.width;
        return this.props.feature.data.map((highlight, i) => {
            let left = util.getPixelFromPosition(+highlight.from / this.props.gui.totalTimeSeconds, 0, width, this.props.gui.zoom);
            let right = util.getPixelFromPosition(+highlight.to / this.props.gui.totalTimeSeconds, 0, width, this.props.gui.zoom);
            if (right < 0 || left > this.props.gui.width) return null;
            const style = { height: "100%", overflow: "hidden" };
            if (highlight.color) Object.assign(style, { backgroundColor: `rgba(${highlight.color.join(",")},0.3)` });
            let className = "highlight";
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
            const padding = 0;
            if (right - left < 5) Object.assign(style, { borderRight: "none", borderLeft: "none" });
            Object.assign(style, { left: left + "px", width: right - left + "px", padding: padding + "px" });
            return <div className={className} key={highlight.from} style={style}>{highlight.text}</div>;
        });
    }
    render() {
        return (
            <div style={{ fontSize: "smaller" }}>{this.getElements()}</div>
        );
    }
}