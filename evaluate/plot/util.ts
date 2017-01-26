import * as mu from 'mobx-utils';
import * as Rx from 'rxjs';
import * as mobx from 'mobx';

const tcache = new WeakMap<Function, {current: any}>();
export function throttleGet<T>(time: number, getter: () => T): T {
    if(tcache.has(getter)) {
        return tcache.get(getter)!.current;
    }
    const ob = Rx.Observable.from<T>(mu.toStream(getter));
    const ob2 = ob.debounceTime(time).merge(ob.delay(time).sampleTime(time));
    const v = mu.fromStream<T>(ob2, mobx.untracked(getter));
    tcache.set(getter, v);
    return v.current;
}