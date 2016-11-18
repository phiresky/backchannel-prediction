import {observer} from 'mobx-react';
import * as React from 'react';

import * as c from './client';
import * as util from './util';

@observer
export class OverlayVisualizer extends React.Component<c.VisualizerProps<c.Feature[]>, {}> {
    render() {
        return (
            <div style={{position: "relative", height:c.globalConfig.visualizerHeight + "px", width: "100%"}}>
                {this.props.feature.map((feature, i) => 
                    <div key={i} style={{position: "absolute", width: "100%", height:"100%"}}>
                        <c.GetVisualizer feature={feature} gui={this.props.gui} uiState={this.props.uiState}/>
                    </div>
                )}
            </div>
        )
    }
}
@observer
export class HighlightsVisualizer extends React.Component<c.VisualizerProps<c.Highlights>, {}> {
    getElements() {
        const width = this.props.gui.width;
        return this.props.feature.data.map((highlight,i) => {
            let left = util.getPixelFromPosition(highlight.from / this.props.gui.totalTimeSeconds, 0, width, this.props.gui.zoom);
            let right = util.getPixelFromPosition(highlight.to / this.props.gui.totalTimeSeconds, 0, width, this.props.gui.zoom);
            if ( right < 0 || left > this.props.gui.width) return null;
            const style = {backgroundColor: `rgba(${highlight.color.join(",")},0.3)`, height: c.globalConfig.visualizerHeight+"px", overflow: "hidden"};
            let className = "highlight";
            if(left < 0) {
                left = 0;
                Object.assign(style, {borderLeft: "none"});
                className += " leftcutoff";
            }
            if(right > width) {
                right = width;
                Object.assign(style, {borderRight: "none"});
                className += " rightcutoff";
            }
            const padding = 0;
            Object.assign(style, {left:left+"px", width: right-left-padding*2+"px", padding: padding + "px"});
            return <div className={className} key={highlight.from} style={style}
                    //onMouseEnter={action("hoverTooltip", _ => this.tooltip = i)}
                    //onMouseLeave={action("hoverTooltipDisable", _ => this.tooltip = null)}
                    >{highlight.text}</div>;
        });
    }
    render() {
        return (
            <div style={{fontSize:"smaller"}}>{this.getElements()}</div>
        )
    }
}