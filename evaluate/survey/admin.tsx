import * as React from 'react';
import { render } from 'react-dom';
import * as mobx from 'mobx';
import { observable } from 'mobx';
import * as mobxReact from 'mobx-react';
import * as io from 'socket.io-client';
import * as common from './common';
import 'whatwg-fetch'
import { Session, NetRating } from './db';
import * as ReactPivot from 'react-pivot';

@mobxReact.observer
class Component extends React.Component<{}, {}> {

}
const initialReduced = {
    ratingTotal: 0,
    ratingCount: 0,
};

type Reduced = typeof initialReduced;

function getPredictor(segment: string) {
    const [, type, ...stuff] = segment.split("/");
    return type;
}
class TPivot extends ReactPivot<NetRating, Reduced> { }
class GUI extends Component {
    @observable.ref
    data: NetRating[] | null = null;
    @observable.ref
    sessions: Session[] | null = null;

    constructor() {
        super();
        this.load();
    }
    async load() {
        let url1 = location.protocol + "//" + location.hostname + ":8001" + location.pathname;
        let url = location.protocol + "//" + location.hostname + ":8001" + location.pathname + "ratings.json";
        url += "?" + Math.random();
        this.data = await (fetch("ratings.json").then(resp => resp.json()) as any);
        this.sessions = await (fetch("sessions.json").then(resp => resp.json()) as any);
    }
    reduce(row: NetRating, reduced: Reduced) {
        if (!reduced.ratingCount) Object.assign(reduced, initialReduced);
        reduced.ratingTotal += row.rating!;
        reduced.ratingCount += 1;
        return reduced;
    }

    render() {
        if (!this.data) return <div>Loading...</div>;
        return (
            <div>
                <button onClick={() => this.load()}>Reload</button>
                <TPivot
                    rows={this.data}
                    dimensions={[
                        { value: row => row.session.id, title: 'Session ID' },
                        { value: row => getPredictor(row.segment), title: 'Predictor' },
                        { value: row => row.ratingType, title: 'Rating Type' },
                        { value: row => !row.final, title: 'wasOverwritten' }
                    ]}
                    reduce={this.reduce}
                    activeDimensions={[
                        "wasOverwritten",
                        "Rating Type",
                        "Predictor"
                    ]}
                    solo={{ title: "wasOverwritten", value: "false" }}
                    calculations={[
                        {
                            title: 'Average Rating',
                            value: row => row.ratingTotal / row.ratingCount,
                            template: (val, row) => {
                                return val.toPrecision(2) + " points";
                            }
                        },
                        {
                            title: 'Rating Count', value: (row: Reduced) => row.ratingCount,
                        }
                    ]}
                />
                <table>
                    <thead>
                        <tr><th>Session ID</th><th>Created</th><th>Comment</th></tr>
                    </thead>
                    <tbody>
                        {this.sessions && this.sessions.map(session => 
                            <tr>
                                <td>{session.id}</td>
                                <td>{session.created}</td>
                                {/*<td>{JSON.stringify(session.handshake, null, 3)}</td>*/}
                                <td>{session.comment}</td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>
        )
    }
}


Object.assign(window, { gui: render(<GUI />, document.getElementById("app")) });
