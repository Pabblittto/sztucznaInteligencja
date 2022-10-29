import { DividerResult } from "./../../utils/datasetDivider";
import * as tf from "@tensorflow/tfjs";

export const createNeuralNetworkModel = async (
  numbOfClasses: number,
  batchSize: number
) => {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [12],
      batchSize: batchSize,
      units: 1,
      activation: "relu",
    })
  ); // Second layer
  model.add(tf.layers.dense({ units: 32, activation: "relu" })); // Second layer
  model.add(tf.layers.dense({ units: 12, activation: "relu" })); // Second layer
  model.add(
    tf.layers.dense({
      units: numbOfClasses,
      batchSize: 1,
      activation: "softmax",
    })
  ); // Second layer

  model.compile({
    optimizer: "adam",
    loss: "meanSquaredError",
  });

  model.summary();
  return model;
};
