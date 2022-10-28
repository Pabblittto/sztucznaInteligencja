import * as tf from "@tensorflow/tfjs";
import { DiscretizatedRowType } from "../../types/baseTypes";

export const createNeuralNetworkModel = async (
  data: tf.data.Dataset<DiscretizatedRowType>
) => {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: [12, 0] })); // First input layer
  model.add(tf.layers.dense({ units: 32, activation: "relu" })); // Second layer
  model.add(tf.layers.flatten({})); // Second layer

  model.summary();

  model.compile();

  //   await model.fit();
};
