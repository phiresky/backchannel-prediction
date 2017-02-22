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
class GUI extends Component {
    @observable.ref
    data: NetRating[] | null = null;

    constructor() {
        super();
        this.load();
    }
    async load() {
        this.data = await (fetch(location.protocol + "//" + location.hostname + ":8001" + location.pathname + "data.json").then(resp => resp.json()) as any);
    }
    reduce(row: NetRating, reduced: Reduced) {
        if (!reduced.ratingCount) Object.assign(reduced, initialReduced);
        reduced.ratingTotal += row.rating!;
        reduced.ratingCount += 1;
        return reduced;
    }

    render() {
        if (!this.data) return <div>Loading...</div>;
        class TPivot extends ReactPivot<NetRating, Reduced> {}
        return (
            <div>
                <TPivot
                    rows={this.data}
                    dimensions={[
                        { value: row => row.session.id, title: 'Session ID' },
                        { value: row => getPredictor(row.segment), title: 'Predictor' },
                        { value: row => row.ratingType, title: 'Rating Type'},
                        { value: row => row.final, title: 'is final'}
                    ]}
                    reduce={this.reduce}
                    calculations={[
                        {
                            title: 'Average Rating',
                            value: row => row.ratingTotal / row.ratingCount,
                            template: (val, row) => {
                                return val + " points";
                            }
                        },
                        {
                            title: 'Rating Count', value: (row: Reduced) => row.ratingCount,
                        }
                    ]}
                />
            </div>
        )
    }
}


Object.assign(window, { gui: render(<GUI />, document.getElementById("app")) });
