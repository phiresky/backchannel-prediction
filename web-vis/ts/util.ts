export function stats(data: number[]) {
    let min = Infinity, max = -Infinity;
    let rms = 0;
    for (const v of data) {
        if (v < min) min = v;
        if (v > max) max = v;
        rms += v * v;
    }
    rms = Math.sqrt(rms / data.length);
    return {min, max, rms};
}

export function getPositionFromPixel(x: number, left: number, width: number, zoom: {left: number, right: number}) {
    let position =  (x - left) / width;
    return (zoom.right - zoom.left) * position + zoom.left;
}

export function getPixelFromPosition(x: number, left: number, width: number, zoom: {left: number, right: number}) {
    return (x - zoom.left) * width / (zoom.right - zoom.left) + left;
}