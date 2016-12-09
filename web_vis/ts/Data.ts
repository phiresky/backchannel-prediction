import * as mobx from "mobx";
import * as util from "./util";

export type TypedArrayType = "float32" | "int16";
function TypedArrayOf(type: TypedArrayType) {
    switch (type) {
        case "float32": return Float32Array;
        case "int16": return Int16Array;
        default: throw Error("unknown type");
    }
}

export type Selector = number | "ALL";
export type FastIterator = {
    start: number;
    count: number;
    stride: number;
}
export type DataIterator = {
    data: TwoDimensionalArray,
    iterator: FastIterator
}
const toReal = (iterator: { count: number, start: number, stride: number }) => (i: number) => {
    if (i > iterator.count || i < 0) throw Error("OOB:" + i + ">=/<=" + iterator.count);
    return iterator.start + i * iterator.stride;
};
const toFake = (iterator: { count: number, start: number, stride: number }, ignoreFloats = false) => (j: number) => {
    const ret = (j - iterator.start) / iterator.stride;
    if (!ignoreFloats && ret !== (ret | 0)) throw Error("ERR1");
    return ret;
};
export class TwoDimensionalArray {
    public readonly buffer: ArrayLike<number>;
    private readonly dataType: TypedArrayType;
    private readonly atom: mobx.Atom;
    readonly shape: [number, number];
    constructor(dataType: TypedArrayType, shape: [number, number]) {
        this.atom = new mobx.Atom("MultiDimensionalArray");
        this.dataType = dataType;
        this.shape = shape;
        this.buffer = new (TypedArrayOf(dataType))(shape.reduce((a, b) => a * b));
    }

    /** iterate over a dimension. one dimension index must be "ALL" */
    iterate(dim1: number | "ALL", dim2: number | "ALL"): FastIterator {
        if (!this.atom.reportObserved()) console.warn("should only access from observer");
        if (dim1 === "ALL" && dim2 !== "ALL") {
            return {
                start: this.getIndex(0, dim2),
                stride: this.shape[1],
                count: this.shape[0]
            };
        } else if (dim1 !== "ALL" && dim2 === "ALL") {
            return {
                start: this.getIndex(dim1, 0),
                stride: 1,
                count: this.shape[1]
            };
        } else throw Error("one dimension must be ALL");
    }
    @mobx.action
    setData(byteOffset: number, buffer: ArrayBuffer, bufferOffset: number, byteLength: number) {
        const Cons = TypedArrayOf(this.dataType);
        const offset = byteOffset / Cons.BYTES_PER_ELEMENT;
        const length = byteLength / Cons.BYTES_PER_ELEMENT;
        const view = new Cons(buffer, bufferOffset, length);
        const backing = this.buffer as Float32Array | Int16Array;
        backing.set(view, byteOffset / Cons.BYTES_PER_ELEMENT);
        this.invalidateRange(offset, offset + length);
        this.atom.reportChanged();
    }
    private invalidateRange(start: number, end: number) {
        for (const [iterator, getter] of this.cache.entries()) {
            if (getter instanceof util.BinaryCacheTree) {
                const _toFake = toFake(iterator, true);
                const fakeStart = _toFake(start), fakeEnd = _toFake(end);
                if (!(fakeEnd < 0 || fakeStart >= iterator.count)) {
                    getter.invalidate(fakeStart, fakeEnd);
                }
            }
        }
    }

    private getIndex(dim1index: number, dim2index: number) {
        return dim1index * this.shape[1] + dim2index;
    }
    cache = new util.LazyHashMap<FastIterator, util.ValueGetter<util.Stats>>();
    public stats(iterator: FastIterator, start: number, end: number) {
        const _toReal = toReal(iterator);
        const getter = {
            getValue: (a: number, b: number) =>
                util.statsRaw(this.buffer, _toReal(a), _toReal(b), iterator.stride)
        };
        let fastGetter = this.cache.get(iterator);
        if (!fastGetter) {
            fastGetter = util.BinaryCacheTree.create(0, iterator.count, getter, util.statsCombinator);
            this.cache.set(iterator, fastGetter);
        }
        return fastGetter.getValue(start, end);
    }
}