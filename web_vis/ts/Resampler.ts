"use strict";
// JavaScript Audio Resampler
// Copyright (C) 2011-2015 Grant Galitz
// Released to Public Domain

export async function nativeResample(inputBuffer: Float32Array, bufferLength: number, fromSampleRate: number, toSampleRate: number) {
    const targetLen = Math.ceil(inputBuffer.length * toSampleRate / fromSampleRate);
    const ctx = new OfflineAudioContext(1, targetLen, toSampleRate);
    const buf = ctx.createBuffer(1, inputBuffer.length, fromSampleRate);
    buf.copyToChannel(inputBuffer, 0);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    src.start(0);
    const result = await ctx.startRendering();
    return result.getChannelData(0);
}
export function linearInterpolate(inputBuffer: Float32Array, bufferLength: number, fromSampleRate: number, toSampleRate: number) {
    const ratioWeight = fromSampleRate / toSampleRate;
    const outputBufferSize = (Math.ceil(inputBuffer.length * toSampleRate / fromSampleRate * 1.000000476837158203125));
    const outputBuffer = new Float32Array(outputBufferSize);
    let lastWeight = 1;
    let outputOffset = 0;
    let lastOutput = 0;
    if (bufferLength > 0) {
        let weight = lastWeight;
        let sourceOffset = 0;
        outputOffset = 0;
        for (; weight < 1; weight += ratioWeight) {
            const secondWeight = weight % 1;
            const firstWeight = 1 - secondWeight;
            outputBuffer[outputOffset++] = (lastOutput * firstWeight) + (inputBuffer[0] * secondWeight);
        }
        weight -= 1;
        for (bufferLength -= 1, sourceOffset = Math.floor(weight) * 1; sourceOffset < bufferLength;) {
            const secondWeight = weight % 1;
            const firstWeight = 1 - secondWeight;
            outputBuffer[outputOffset++] = (inputBuffer[sourceOffset] * firstWeight) + (inputBuffer[sourceOffset + 1] * secondWeight);
            weight += ratioWeight;
            sourceOffset = Math.floor(weight) * 1;
        }
        lastOutput = inputBuffer[sourceOffset++];
        lastWeight = weight % 1;
    }
    if (outputOffset !== outputBufferSize - 1) throw Error(outputOffset + "≠" + outputBufferSize);
    return outputBuffer;
}
export function multiTapInterpolate(inputBuffer: Float32Array, bufferLength: number, fromSampleRate: number, toSampleRate: number) {
    let tailExists = false;
    let lastWeight = 0;
    const ratioWeight = fromSampleRate / toSampleRate;
    const outputBufferSize = (Math.ceil(inputBuffer.length * toSampleRate / fromSampleRate * 1.000000476837158203125));
    const outputBuffer = new Float32Array(outputBufferSize);
    let lastOutput = 0;
    let outputOffset = 0;
    if (bufferLength > 0) {
        let weight = 0;
        let output0 = 0;
        let actualPosition = 0;
        let amountToNext = 0;
        let alreadyProcessedTail = !tailExists;
        tailExists = false;
        let currentPosition = 0;
        do {
            if (alreadyProcessedTail) {
                weight = ratioWeight;
                output0 = 0;
            } else {
                weight = lastWeight;
                output0 = lastOutput;
                alreadyProcessedTail = true;
            }
            while (weight > 0 && actualPosition < bufferLength) {
                amountToNext = 1 + actualPosition - currentPosition;
                if (weight >= amountToNext) {
                    output0 += inputBuffer[actualPosition++] * amountToNext;
                    currentPosition = actualPosition;
                    weight -= amountToNext;
                } else {
                    output0 += inputBuffer[actualPosition] * weight;
                    currentPosition += weight;
                    weight = 0;
                    break;
                }
            }
            if (weight <= 0) {
                outputBuffer[outputOffset++] = output0 / ratioWeight;
            } else {
                lastWeight = weight;
                lastOutput = output0;
                tailExists = true;
                break;
            }
        } while (actualPosition < bufferLength);
    }
    if (outputOffset !== outputBufferSize - 1) throw Error(outputOffset + "≠" + outputBufferSize);
    return outputBuffer;
}