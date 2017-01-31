import { Line } from 'react-chartjs-2';
import * as React from 'react';
import * as ReactDOM from 'react-dom';
import { Table } from 'reactable';
import { observable, action, computed } from 'mobx';
import * as mobx from 'mobx';
import { observer, Observer } from 'mobx-react';
import 'react-select/dist/react-select.css';
import * as Select from 'react-select';
import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
import * as util from './util';
import * as mu from 'mobx-utils';
import * as Rx from 'rxjs';


type Column<T> = {
    name: string;
    sort?: {
        direction: "ascending" | "descending";
        priority: 0;
    },
} & ({
    type: "number",
    filter?: {
        min: number; max: number
    },
    selector: (t: T) => number;
} | {
        type: "categorical",
        filter?: {
            values: string[]
        }
        selector: (t: T) => string;
    } | {
        type: "freetext",
        filter?: {
            regex: string
        }
        selector: (t: T) => string;
    }
    )
type Filter<T> = (element: T, column: Column<T>) => boolean;

export interface Props<T> {
    data: T[];
    columns: { [name: string]: Column<T> },
    visibleColumns: string[];
}

function Sortie({ column }: { column: Column<any> }) {
    if (!column.sort) return <span />;
    return <p>Sorted: {column.sort.direction} (prio {column.sort.priority})</p>;
}
const possibleValues = mobx.createTransformer(
    function possibleValues<T>(info: { data: T[], column: Column<T> }): Set<string> {
        const possible = new Set();
        for (const point of info.data) possible.add(info.column.selector(point) as string);
        return possible;
    });
function Filtri<T>({ column, data }: { column: Column<T>, data: T[] }) {
    let select;
    switch (column.type) {
        case "categorical": {
            if (!column.filter) column.filter = { values: [] };
            const filter = column.filter;
            select = <Select
                multi
                value={column.filter.values.map(value => ({ value, label: value }))}
                options={[...possibleValues({ data, column })].map(value => ({ value, label: value }))}
                onChange={(v: Select.Option[]) => filter.values = v.map(option => option.value as string)}
            />;
            break;
        }
        case "number": {
            if (!column.filter) column.filter = { min: -Infinity, max: Infinity };
            const filter = column.filter;
            select = (
                <div>
                    <label>Min<input type="number" value={filter.min} onChange={e => filter.min = +e.currentTarget.value} /></label>
                    <label>Max<input type="number" value={filter.max} onChange={e => filter.max = +e.currentTarget.value} /></label>
                </div>
            );
            break;
        }
        case "freetext": {
            if (!column.filter) column.filter = { regex: "" };
            const filter = column.filter;
            select = (
                <label>Regex
                    <input value={filter.regex} onChange={e => filter.regex = e.currentTarget.value} />
                </label>
            )
            break;
        }
    }
    return (
        <div>
            {select}
        </div>
    );
}
function getPredicate<T>(column: Column<T>): (t: T) => boolean {
    switch (column.type) {
        case "freetext":
            try {
                const regex = RegExp(column.filter!.regex);
                return (t: T) => column.selector(t).search(regex) >= 0;
            } catch (e) {
                return (t: T) => true;
            }
        case "number":
            const {min, max} = column.filter!;
            return (t: T) => {
                const val = column.selector(t);
                return min <= val && val <= max;
            }
        case "categorical":
            return (t: T) => column.filter!.values.includes(column.selector(t));
    }
}
function filter<T>(data: T[], columns: Column<T>[]) {
    const filterCols = columns.filter(column => column.filter).map(getPredicate);
    const filterFn = (data: T) => {
        return filterCols.every(filter => filter(data));
    }
    return data.filter(filterFn);
}
@observer
export class UltimateTable<T> extends React.Component<Props<T>, {}> {
    @computed get columnNames() {
        return new Set(Object.keys(this.props.columns))
    }
    //createTransformer(function)
    render() {
        const p = this.props;
        const allColumns = new Set(Object.keys(p.columns));
        const invisibleColumns = new Set(allColumns);
        for (const column in p.visibleColumns) invisibleColumns.delete(column);
        const MyFiltri = (props: { column: Column<T>, data: T[] }) => Filtri(props);
        const data = filter(p.data, Object.values(p.columns));
        return (
            <div>
                <table className="ultimate-table">
                    <thead>
                        <tr>
                            {p.visibleColumns.map(columnName =>
                                <th key={columnName}>
                                    <p>{columnName}</p>
                                    <Sortie column={p.columns[columnName]} />
                                    <MyFiltri column={p.columns[columnName]} data={p.data} />
                                </th>
                            )}
                        </tr>
                    </thead>
                    <tbody>
                        {data.map((entry,i) => (
                            <tr key={i}>
                                {p.visibleColumns.map(name => 
                                    <td key={name}>
                                        {p.columns[name].selector(entry)}
                                    </td>
                                )}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        )
    }
}