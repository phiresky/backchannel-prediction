import * as mu from 'mobx-utils';
import * as Rx from 'rxjs';
import * as mobx from 'mobx';

const tcache = new WeakMap<Function, {current: any}>();
export function throttleGet<T>(time: number, getter: () => T): T {
    if(tcache.has(getter)) {
        return tcache.get(getter)!.current;
    }
    const ob = Rx.Observable.from<T>(mu.toStream(getter));
    const ob2 = ob.debounceTime(time); //.merge(ob.delay(time).sampleTime(time));
    const v = mu.fromStream<T>(ob2, mobx.untracked(getter));
    tcache.set(getter, v);
    return v.current;
}

export function copyToClipboard(text: string) {
    if (document.queryCommandSupported && document.queryCommandSupported("copy")) {
        var textarea = document.createElement("textarea");
        textarea.textContent = text;
        textarea.style.position = "fixed";  // Prevent scrolling to bottom of page in MS Edge.
        document.body.appendChild(textarea);
        textarea.select();
        try {
            return document.execCommand("copy");  // Security exception may be thrown by some browsers.
        } catch (ex) {
            console.warn("Copy to clipboard failed.", ex);
            return false;
        } finally {
            document.body.removeChild(textarea);
        }
    } else {
        console.error("copy not supported");
        return false;
    }
}