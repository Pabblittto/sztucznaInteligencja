import * as tf from "@tensorflow/tfjs";
import * as fs from "fs/promises";
import path from "path";
import { discretization } from "./utils/discretization";

async function main() {
  const dataLocation = "file://" + __dirname + "/data/forestfires.csv";

  const columnConfigs: {
    [key: string]: tf.data.ColumnConfig;
  } = {
    X: { dtype: "int32" },
    Y: { dtype: "int32" },
    month: { dtype: "string" },
    day: { dtype: "string" },
    FFMC: { dtype: "float32" },
    DMC: { dtype: "float32" },
    ISI: { dtype: "float32" },
    temp: { dtype: "float32" },
    RH: { dtype: "int32" },
    wind: { dtype: "float32" },
    rain: { dtype: "float32" },
    area: { dtype: "float32" },
  };

  const csvDataset = tf.data.csv(dataLocation, {
    hasHeader: true,
    columnConfigs: columnConfigs,
  });

  discretization(csvDataset);
}

main();
