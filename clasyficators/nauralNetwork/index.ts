import { Optimizer, LossFunction } from "./../../types/baseTypes";
import { DividerResult } from "./../../utils/datasetDivider";
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
      inputShape: [12],
      batchSize: 1,
      units: 1,
      activation: "relu",
    })
  ); // Second layer
  model.add(tf.layers.dense({ units: 24, activation: "relu" })); // Second layer
  model.add(tf.layers.dense({ units: 64, activation: "relu" })); // Second layer
  model.add(tf.layers.dense({ units: 32, activation: "relu" })); // Second layer
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
    optimizer: optimizer,
    loss: loss,
    metrics: ["accuracy"],
  });

  model.summary();
  return model;
};
