import { EntireDatasetValidationResult } from "../../types/validationResult";
import { datasetDivider } from "../../utils/datasetDivider";
import {
  Optimizer,
  LossFunction,
  ClearedRowData,
} from "./../../types/baseTypes";
import * as tf from "@tensorflow/tfjs";

export const createNeuralNetworkModel = async (
  numbOfClasses: number,
  batchSize: number,
  optimizer: Optimizer,
  loss: LossFunction
) => {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [7],
      batchSize: 1,
      units: 1,
      activation: "relu",
    })
  ); // Second layer
  model.add(tf.layers.dense({ units: 14, activation: "relu" })); // Second layer
  model.add(tf.layers.dense({ units: 28, activation: "relu" })); // Second layer
  model.add(tf.layers.dense({ units: 28, activation: "relu" })); // Second layer
  model.add(tf.layers.dense({ units: 14, activation: "relu" })); // Second layer
  model.add(tf.layers.dense({ units: 7, activation: "relu" })); // Second layer
  model.add(
    tf.layers.dense({
      units: numbOfClasses,
      batchSize: 1,
      activation: "softmax",
    })
  ); // Second layer

  model.compile({
    optimizer: optimizer,
    loss: loss,
    metrics: ["accuracy"],
  });

  model.summary();
  return model;
};

/**
 * Returns index of cell where was the largest number was contained
 */
const defineTheLargestValueInArray = (array: number[]) => {
  let maxIndex = 0;
  let max = 0;
  array.forEach((val, index) => {
    if (val > max) {
      max = val;
      maxIndex = index;
    }
  });
  return maxIndex;
};

export const validateModel = async (
  model: tf.Sequential,
  dataSet: tf.data.Dataset<ClearedRowData>,
  numberOfClasses: number
): Promise<EntireDatasetValidationResult> => {
  const result: EntireDatasetValidationResult = new Array(5);

  for (let i = 0; i < result.length; i++) {
    result[i] = new Array(5).fill(0);
  }

  const arrayData = (await dataSet.toArray()).map(
    async (x) => await datasetDivider(x, numberOfClasses)
  );

  const predictArgs = await Promise.all(arrayData);

  predictArgs.forEach((row) => {
    const realGroupIndex = defineTheLargestValueInArray(row.ys.arraySync());
    const prediction = model.predict(row.xs.reshape([1, 7]));
    if (!Array.isArray(prediction)) {
      const predictedGroupIndex = defineTheLargestValueInArray(
        prediction.arraySync() as number[]
      );
      result[realGroupIndex][predictedGroupIndex] += 1;
    } else {
      throw "Shouldn't get here";
    }
  });
  console.log("===================");
  console.log("Error Matrix:");
  console.log(result);

  return result;
};
