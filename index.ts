import { datasetDivider } from "./utils/datasetDivider";
import * as tf from "@tensorflow/tfjs";
import * as fs from "fs/promises";
import path from "path";
import { createNeuralNetworkModel } from "./clasyficators/nauralNetwork";
import { columnConfigs } from "./types/baseTypes";
import { getNormalizingFunction, NormalizationType } from "./utils/normalize";
import { saveDatasetAsCsv } from "./utils/saveDatasetAsCsv";
import { getBatchSize } from "./utils/getBatchSize";

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

  const batchSize: number | undefined = undefined; // Batch size, set to undefined if you want to use default batch size

  // SETTINGS =========================================================================

  // Prepare data:
  const csvDataset = tf.data
    .csv(dataLocation, {
      hasHeader: true,
      columnConfigs: columnConfigs,
    })
    .mapAsync(getNormalizingFunction(normalizeType));

  const finalBatchSize = getBatchSize(
    (await csvDataset.toArray()).length,
    batchSize
  );

  // TODO: zrobić pipeline
  // TODO: określić moze inne typy dyskretyzacji, na przykład liniowy podział zakresu, albo logarytmiczny
  // TODO:
  // trzeba sprawdzić czy nie ma dublikatów tzn. wierszy o takich samych wartościach, ale o innej wartości zmiennej decyzyjnej

  // TODO: tu jakieś drzewa decysyjne + sieci neuronowe, jakieś pojebane coś
  // Save clear data to file
  await saveDatasetAsCsv(`cleared_${normalizeType}`, csvDataset);

  const dividedData = csvDataset.mapAsync(datasetDivider).batch(finalBatchSize);

  // Model creating
  const model = await createNeuralNetworkModel(numbOfClasses, finalBatchSize);

  // Training
  await model.fitDataset(dividedData, {
    epochs: 200,
    verbose: 0,
    callbacks: {
      onEpochEnd(epoch, logs?) {
        if (logs) {
          console.log(`==========`);
          console.log(
            `train-set, epoch: ${epoch} loss: ${logs.loss.toFixed(4)}`
          );
        }
      },
    },
  });

  console.log(model.outputs);
}

main();
