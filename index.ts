import * as tf from "@tensorflow/tfjs";
import * as fs from "fs/promises";
import path from "path";
import { createNeuralNetworkModel } from "./clasyficators/nauralNetwork";
import { discretization } from "./utils/discretization";
import { saveDatasetAsCsv } from "./utils/saveDatasetAsCsv";

async function main() {
  const dataLocation = "file://" + __dirname + "/data/forestfires.csv"; // MAC OS
  // const dataLocation = "file://" + __dirname + "\\data\\forestfires.csv"; // Windows

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

  // TODO: zrobić pipeline

  const cvsClearedData = await discretization(csvDataset);
  // TODO: określić moze inne typy dyskretyzacji, na przykład liniowy podział zakresu, albo logarytmiczny
  // TODO:
  // trzeba sprawdzić czy nie ma dublikatów tzn. wierszy o takich samych wartościach, ale o innej wartości zmiennej decyzyjnej

  // TODO: tu jakieś drzewa decysyjne + sieci neuronowe, jakieś pojebane coś
  // Save clear data to file
  await saveDatasetAsCsv("cleared", cvsClearedData);

  // Model creating

  await createNeuralNetworkModel(cvsClearedData);
}

main();
