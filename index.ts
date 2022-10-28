import { datasetDivider } from "./utils/datasetDivider";
import * as tf from "@tensorflow/tfjs";
import * as fs from "fs/promises";
import path from "path";
import { createNeuralNetworkModel } from "./clasyficators/nauralNetwork";
import { columnConfigs } from "./types/baseTypes";
import { getNormalizingFunction, NormalizationType } from "./utils/normalize";
import { saveDatasetAsCsv } from "./utils/saveDatasetAsCsv";

async function main() {
  // SETTINGS ==================================================================

  const dataLocation =
    process.platform === "win32"
      ? "file://" + __dirname + "\\data\\forestfires.csv"
      : "file://" + __dirname + "/data/forestfires.csv"; // DAta location link based on OS os Dev

  /**
   * Normalize type - define how data should be normalized
   */
  const normalizeType: NormalizationType = NormalizationType.EXPERT;

  const numbOfClasses = 5; // 5 number of calsses because ther is 5 types of area. See:  AreaThresholds

  // SETTINGS =========================================================================

  // Prepare data:
  const csvDataset = tf.data
    .csv(dataLocation, {
      hasHeader: true,
      columnConfigs: columnConfigs,
    })
    .mapAsync(getNormalizingFunction(normalizeType));

  // TODO: zrobić pipeline
  // TODO: określić moze inne typy dyskretyzacji, na przykład liniowy podział zakresu, albo logarytmiczny
  // TODO:
  // trzeba sprawdzić czy nie ma dublikatów tzn. wierszy o takich samych wartościach, ale o innej wartości zmiennej decyzyjnej

  // TODO: tu jakieś drzewa decysyjne + sieci neuronowe, jakieś pojebane coś
  // Save clear data to file
  await saveDatasetAsCsv(`cleared_${normalizeType}`, csvDataset);

  const dividedData = csvDataset.mapAsync(datasetDivider);

  // Model creating
  const model = await createNeuralNetworkModel(numbOfClasses);

  // Training

  model.fitDataset(dividedData, {
    epochs: 10,
    verbose: 0,
    callbacks: {
      onEpochEnd(epoch, logs?) {
        if (logs) console.log(`train-set loss: ${logs.loss.toFixed(4)}`);
      },
    },
  });
}

main();
