import * as Data from "./Data";

export type NumFeatureCommon = {
    name: FeatureID,
    samplingRate: number, // in kHz
    shift: number,
    data: Data.TwoDimensionalArray,
    range: [number, number] | null
};
export type NumFeatureSVector = NumFeatureCommon & {
    typ: "FeatureType.SVector", dtype: "int16"
};

export type NumFeatureFMatrix = NumFeatureCommon & {
    typ: "FeatureType.FMatrix", dtype: "float32"
};
export type Color = [number, number, number];

export type Utterances = {
    name: FeatureID,
    typ: "utterances",
    data: Utterance[]
}
export type Highlights = {
    name: FeatureID,
    typ: "highlights",
    data: Utterance[]
}
export type NumFeature = NumFeatureSVector | NumFeatureFMatrix;
export type Feature = NumFeature | Utterances | Highlights;
export type Utterance = { from: number | string, to: number | string, text?: string, id?: string, color?: Color };

export interface ConversationID extends String {
    __typeBrand: "ConversationID";
}
export interface FeatureID extends String {
    __typeBrand: "FeatureID";
}
export function isFeatureID(f: any): f is FeatureID {
    return typeof f === "string";
}