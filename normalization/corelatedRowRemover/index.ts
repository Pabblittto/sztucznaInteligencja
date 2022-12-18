import * as tf from "@tensorflow/tfjs";
import { ClearedRowData, DiscretizatedRowType } from "../../types/baseTypes";

export const correlatedRowRemover = async (
  row: tf.TensorContainer
): Promise<ClearedRowData> => {
  const typedRow: DiscretizatedRowType = row as DiscretizatedRowType;
  const { month, FFMC, DMC, DC, ISI, ...rest } = typedRow;
  return rest;
};
