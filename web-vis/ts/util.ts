type ValueGetter<T> = (start: number, end: number, evaluator: (start: number, end: number) => T, combinator: (t1: T, t2: T) => T) => T;

const limit = 32;
class BinaryCacheTree<T> {
    private value: T;
    private left: ValueGetter<T>;
    private right: ValueGetter<T>;
    constructor(private start: number, private end: number,
                evaluator: (start: number, end: number) => T, combinator: (t1: T, t2: T) => T) {
        const mid = Math.floor((start + end) / 2);
        this.left = BinaryCacheTree.create(start, mid, evaluator, combinator);
        this.right = BinaryCacheTree.create(mid, end, evaluator, combinator);
        this.value = combinator(this.left(start, mid, evaluator, combinator), this.right(mid, end, evaluator, combinator));
    }
    static create<T>(start: number, end: number, evaluator: (start: number, end: number) => T, combinator: (t1: T, t2: T) => T): ValueGetter<T> {
        const length = end - start;
        if(length < limit) return evaluator;
        else {
            const bct = new BinaryCacheTree(start, end, evaluator, combinator);
            return bct.getValue.bind(bct);
        }
    }
    getValue(start: number, end: number, evaluator: (start: number, end: number) => T, combinator: (t1: T, t2: T) => T): T {
        if (end - start < limit) return evaluator(start, end);
        if (start < this.start || end > this.end) throw Error("Out of range");
        if (start === this.start && end === this.end) {
            return this.value;
        }
        let value1, value2;
        const mid = Math.floor((this.start + this.end) / 2);
        if (start < mid) value1 = this.left(start, Math.min(mid, end), evaluator, combinator);
        if (end > mid) value2 = this.right(Math.max(start, mid), end, evaluator, combinator)!;
        if (value1 === undefined && value2 === undefined) throw Error("weird?");
        else if (value1 === undefined && value2 !== undefined) return value2;
        else if (value2 === undefined && value1 !== undefined) return value1;
        else return combinator(value1!, value2!);
    }
}

export function statsRaw(data: number[], start: number, end: number): Stats {
    let min = Infinity, max = -Infinity;
    let rms2 = 0;
    let sum = 0;
    for (let i = start; i < end; i++) {
        const v = data[i];
        if (v < min) min = v;
        if (v > max) max = v;
        rms2 += v * v;
        sum += v;
    }
    rms2 = rms2 / (end - start);
    return {min, max, rms2, sum, count: end - start};
}

type Stats = {min: number, max: number, rms2: number, sum: number, count: number};
const cache = new Map<number[], ValueGetter<Stats>>();
const statsCombinator = (stats1: Stats, stats2: Stats) => ({
    min: Math.min(stats1.min, stats2.min),
    max: Math.max(stats1.max, stats2.max),
    rms2: (stats1.count * stats1.rms2 + stats2.count * stats2.rms2) / (stats1.count + stats2.count),
    count: stats1.count + stats2.count,
    sum: stats1.sum + stats2.sum
});
export function stats(data: number[], start: number, end: number): Stats {
    if(!cache.has(data))
        cache.set(data, BinaryCacheTree.create(0, data.length, (start, end) => statsRaw(data, start, end), statsCombinator));
    return cache.get(data)!(start, end, (start, end) => statsRaw(data, start, end), statsCombinator);
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