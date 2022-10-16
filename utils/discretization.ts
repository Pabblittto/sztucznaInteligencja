import * as tf from "@tensorflow/tfjs";
import { BaseRowType } from "../types/baseTypes";

export const discretization = (data: tf.data.CSVDataset) => {
  const discreditedData = data.mapAsync(async (row) => {
    return row;
  });

  return discreditedData;
};
