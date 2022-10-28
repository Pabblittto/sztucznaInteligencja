import * as tf from "@tensorflow/tfjs";
import { DiscretizatedRowType } from "../types/baseTypes";

/**
 * Rozdziela zbiór danych na dwa tensery. Jeden z danymi wejściowymi i drugi z danymi wyjściowymi (area)
 */
export const datasetDivider = async (
  data: tf.data.Dataset<DiscretizatedRowType>
) => {};
