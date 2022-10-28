import { DividerResult } from "./../../utils/datasetDivider";
import * as tf from "@tensorflow/tfjs";

export const createNeuralNetworkModel = async (numbOfClasses: number) => {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({ inputShape: [12], units: 32, activation: "relu" })
  ); // Second layer
  model.add(tf.layers.dense({ units: 64, activation: "relu" })); // Second layer
  model.add(tf.layers.dense({ units: 32, activation: "relu" })); // Second layer
  model.add(tf.layers.dense({ units: 12, activation: "relu" })); // Second layer
  model.add(
    tf.layers.dense({
      units: 12,
      batchSize: numbOfClasses,
      activation: "softmax",
    })
  ); // Second layer

  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
  });

  model.summary();
  return model;
};
