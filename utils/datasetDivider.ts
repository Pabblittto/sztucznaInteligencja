import * as tf from "@tensorflow/tfjs";
import {
  BaseRowType,
  ClearedRowData,
  DiscretizatedRowType,
} from "../types/baseTypes";

export type DividerResult = {
  xs: tf.Tensor<tf.Rank.R1>;
  ys: tf.Tensor<tf.Rank.R1>;
};

/**
 * Rozdziela zbiór danych na dwa tensery. Jeden z danymi wejściowymi i drugi z danymi wyjściowymi (area)
 */
export const datasetDivider = async (
  row: ClearedRowData,
  numberOfClasses: number
): Promise<DividerResult> => {
  const typedRow = row;

  const { area, ...d } = typedRow; // rozdziel zmienną decyzujną od reszty danych

  const zeros = new Array(numberOfClasses).fill(0);

  return {
    xs: tf.tensor<tf.Rank.R1>([d.X, d.Y, d.day, d.RH, d.rain, d.temp, d.wind]),
    ys: tf.tensor<tf.Rank.R1>(zeros.map((z, i) => (i === area ? 1 : 0))), // Creates One-hot array, so if array is "small", the output is: [0,1,0,0,0]
  };
};

export const divideDataset = async (
  dataset: tf.data.Dataset<ClearedRowData>,
  numbOfClasses: number
): Promise<DividerResult[]> => {
  return dataset.mapAsync((x) => datasetDivider(x, numbOfClasses)).toArray();
};
