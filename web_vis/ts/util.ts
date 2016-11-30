import {VisualizerConfig} from './client';
import * as Data from './Data';

export interface ValueGetter<T> {
    getValue : (start: number, end: number) => T;
}

const limit = 32;
export class BinaryCacheTree<T> implements ValueGetter<T> {
    private value: T;
    private left: ValueGetter<T>;
    private right: ValueGetter<T>;
    constructor(private start: number, private end: number,
                private evaluator: ValueGetter<T>, private combinator: (t1: T, t2: T) => T) {
        const mid = Math.floor((start + end) / 2);
        this.left = BinaryCacheTree.create(start, mid, evaluator, combinator);
        this.right = BinaryCacheTree.create(mid, end, evaluator, combinator);
        this.value = combinator(this.left.getValue(start, mid), this.right.getValue(mid, end));
    }
    static create<T>(start: number, end: number, evaluator: ValueGetter<T>, combinator: (t1: T, t2: T) => T): ValueGetter<T> {
        const length = end - start;
        if(length < limit) return evaluator;
        else {
            return new BinaryCacheTree(start, end, evaluator, combinator);
        }
    }
    static toValueGetter<T>(b: BinaryCacheTree<T> | ValueGetter<T>) {
        if(typeof b === 'function') return b;
        else return b.getValue.bind(b);
    }
    getValue(start: number, end: number): T {
        // short cut if we know the caller only wants a short range
        if (end - start < limit) return this.evaluator.getValue(start, end);
        if (start < this.start || end > this.end) throw Error("Out of range");
        if (start === this.start && end === this.end) {
            return this.value;
        }
        let value1, value2;
        const mid = Math.floor((this.start + this.end) / 2);
        if (start < mid) value1 = this.left.getValue(start, Math.min(mid, end));
        if (end > mid) value2 = this.right.getValue(Math.max(start, mid), end)!;
        if (value1 === undefined && value2 === undefined) throw Error("weird?");
        else if (value1 === undefined && value2 !== undefined) return value2;
        else if (value2 === undefined && value1 !== undefined) return value1;
        else return this.combinator(value1!, value2!);
    }
    invalidate(start: number, end: number) {
        const mid = Math.floor((this.start + this.end) / 2);
        if (start < mid) this.left instanceof BinaryCacheTree && this.left.invalidate(start, Math.min(mid, end));
        if (end > mid) this.right instanceof BinaryCacheTree && this.right.invalidate(Math.max(start, mid), end);
        this.value = this.combinator(this.left.getValue(this.start, mid), this.right.getValue(mid, this.end));
    }
}

export function statsRaw(data: ArrayLike<number>, start: number, end: number, stride = 1): Stats {
    let min = Infinity, max = -Infinity;
    let rms2 = 0;
    let sum = 0;
    let count = 0;
    for (let i = start; i < end; i += stride) {
        const v = data[i];
        if (v < min) min = v;
        if (v > max) max = v;
        rms2 += v * v;
        sum += v;
        count++;
    }
    rms2 = rms2 / count;
    return {min, max, rms2, sum, count};
}

export type Stats = {min: number, max: number, rms2: number, sum: number, count: number};
export const cache = new Map<ArrayLike<number>, ValueGetter<Stats>>();
export function statsCombinator(stats1: Stats, stats2: Stats) {
    const count = stats1.count + stats2.count;
    return {
        min: Math.min(stats1.min, stats2.min),
        max: Math.max(stats1.max, stats2.max),
        rms2: (stats1.count * stats1.rms2 + stats2.count * stats2.rms2) / count,
        count, sum: stats1.sum + stats2.sum
    }
}
export function stats(data: ArrayLike<number>, start: number, end: number): Stats {
    if(!cache.has(data))
        cache.set(data, BinaryCacheTree.create(0, data.length, {getValue: (start, end) => statsRaw(data, start, end)}, statsCombinator));
    return cache.get(data)!.getValue(start, end);
}

export function getPositionFromPixel(x: number, left: number, width: number, zoom: {left: number, right: number}) {
    let position =  (x - left) / width;
    return (zoom.right - zoom.left) * position + zoom.left;
}

export function getPixelFromPosition(x: number, left: number, width: number, zoom: {left: number, right: number}) {
    return (x - zoom.left) * width / (zoom.right - zoom.left) + left;
}

export function binarySearch<T>(min: number, max: number, extractor: (i: number) => number, searchValue: number): number|null {
    if (max - min === 0) return null;
    if(max - min === 1) return min;
    const mid = ((max + min) / 2)|0;
    const midVal = extractor(mid);
    if(midVal < searchValue) return binarySearch(mid, max, extractor, searchValue);
    else return binarySearch(min, mid, extractor, searchValue);
}

export function getMinMax(givenRange: [number, number]|null, config: VisualizerConfig, data: Data.DataIterator, start: number, end: number): {min: number, max: number} {
    if(config === "normalizeGlobal") {
        const {min, max} = data.data.stats(data.iterator, 0, data.iterator.count);
        return {min, max};
    }
    else if(config === "normalizeLocal") {
        const {min, max} = data.data.stats(data.iterator, start, end);
        return {min, max};
    }
    else if(config === "givenRange") return givenRange?{min: givenRange[0], max: givenRange[1]}:{min:0, max:1};
    else throw Error("unknown config "+config);
}
export function round1(num: number) {
    if(+num.toPrecision(4) === (num|0)) return num;
    else return num.toPrecision(4);
}
export function randomChoice<T>(data: T[]) {
    return data[Math.floor(Math.random() * data.length)];
}
/**
 * {b:1, a:2} => [["a", 2], ["b", 1]]
 * [1,2,3] => [null, [1, 2, 3]]
 * "hi" => "hi"
 */
export function toDeterministic(obj: any): any {
    if(obj instanceof Array) return [null, obj.map(x => toDeterministic(x))];
    if(obj instanceof Object) return Object.keys(obj).sort().map(k => [k, toDeterministic(obj[k])])
    return obj;
}
export function fromDeterministic(obj: any): any {
    if(obj instanceof Array) {
        if(obj[0] === null) return obj[1];
        const res = {} as any;
        for(const [key, value] of obj) res[key] = fromDeterministic(value);
        return res;
    } else return obj;
}

export class LazyHashMap<K, V> {
    private readonly map = new Map<string, V>();
    set(k: K, v: V) { return this.map.set(JSON.stringify(toDeterministic(k)), v); }
    get(k: K) { return this.map.get(JSON.stringify(toDeterministic(k))); }
    clear() { return this.map.clear(); }
    delete(k: K) { return this.map.delete(JSON.stringify(toDeterministic(k))); }
    has(k: K) { return this.map.has(JSON.stringify(toDeterministic(k))); }
    *entries(): IterableIterator<[K, V]> {
        for(const [k,v] of this.map.entries()) yield [fromDeterministic(JSON.parse(k)), v]; 
    }
}

export function rescale({left = 0, right = 1}, scaleChange: number, position: number) {
    left -= position;
    right -= position;
    left *= scaleChange; 
    right *= scaleChange;
    left += position;
    right += position;
    return {left, right};
}