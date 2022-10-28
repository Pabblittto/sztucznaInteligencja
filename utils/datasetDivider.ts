import * as tf from "@tensorflow/tfjs";
import { BaseRowType, DiscretizatedRowType } from "../types/baseTypes";

export type DividerResult = {
  xs: tf.Tensor;
  ys: tf.Tensor;
};

/**
 * Rozdziela zbiór danych na dwa tensery. Jeden z danymi wejściowymi i drugi z danymi wyjściowymi (area)
 */
export const datasetDivider = async (
  row: tf.TensorContainer
): Promise<DividerResult> => {
  const typedRow: BaseRowType = row as BaseRowType;

  const { area, ...d } = typedRow; // rozdziel zmienną decyzujną od raszy danych

  return {
    xs: tf.tensor([
      d.X,
      d.Y,
      d.day,
      d.month,
      d.FFMC,
      d.DMC,
      d.DC,
      d.ISI,
      d.RH,
      d.rain,
      d.temp,
      d.wind,
    ]),
    ys: tf.tensor([area]),
  };
};
