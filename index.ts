import { datasetDivider } from "./utils/datasetDivider";
import * as tf from "@tensorflow/tfjs";
import {
  createNeuralNetworkModel,
  validateModel,
} from "./clasyficators/nauralNetwork";
import { columnConfigs, Optimizer, LossFunction } from "./types/baseTypes";
import { getNormalizingFunction, NormalizationType } from "./utils/normalize";
import { saveDatasetAsCsv } from "./utils/saveDatasetAsCsv";
import { getBatchSize } from "./utils/getBatchSize";
import "tfjs-node-save"; // very important line!
import { DuplicateDealer } from "./normalization/dublicates";
import { correlatedRowRemover } from "./normalization/corelatedRowRemover";
import { createErrorMatrixFile } from "./utils/createErrorMatrixFile";
import { ReasampleTool } from "./normalization/reasampleTool";

async function main() {
  // SETTINGS ==================================================================
  const dataLocation =
    process.platform === "win32"
      ? "file://" + __dirname + "\\data\\forestfires.csv"
      : "file://" + __dirname + "/data/forestfires.csv"; // DAta location link based on OS os Dev

  const saveModelLocation =
    process.platform === "win32"
      ? "file://" + __dirname + "\\savedModels"
      : "file://" + __dirname + "/savedModels"; // Direcotry path to place where created models are saved

  const saveErrorMatrixFileName = "errorMatrix.html"; // File name of error matrix file

  /**
   * Normalize type - define how data should be normalized
   */
  const normalizeType: NormalizationType = NormalizationType.EXPERT;

  const numbOfClasses = 5; // 5 number of calsses because ther is 5 types of area. See:  AreaThresholds

  const batchSize: number | undefined = undefined; // Batch size, set to undefined if you want to use default batch size
  const epochs = 3000; // Number of learning times
  // For Optimizer and Loss function see: https://www.tensorflow.org/js/guide/train_models#optimizer_loss_and_metric
  const optimizer: Optimizer = Optimizer.rmsprop;
  const lossFunction: LossFunction = LossFunction.meanSquaredError;
  // SETTINGS =========================================================================

  // Prepare data:
  const csvDataset = tf.data
    .csv(dataLocation, {
      hasHeader: true,
      columnConfigs: columnConfigs,
    })
    .mapAsync(getNormalizingFunction(normalizeType)) //! NORMALIZATION
    .mapAsync(correlatedRowRemover); //! DELETE MOST CORRELATED ROWS

  const clearedData = await new DuplicateDealer().dealWithDuplicates(
    csvDataset
  ); //! DELETE DUPLICATES

  const finalDataset = await ReasampleTool.reasample(
    clearedData,
    120,
    numbOfClasses
  ); // ! REBALANCING DATASET

  const finalBatchSize = getBatchSize(
    (await finalDataset.toArray()).length,
    batchSize
  );

  // TODO: grid search - to jest wane
  // TODO: DODAĆ KNN klasyfikator
  // Save clear data to file
  await saveDatasetAsCsv(`cleared_${normalizeType}`, finalDataset);

  const dividedData = finalDataset
    .mapAsync((x) => datasetDivider(x, numbOfClasses))
    .batch(finalBatchSize);

  // Model creating
  const model = await createNeuralNetworkModel(
    numbOfClasses,
    finalBatchSize,
    optimizer,
    lossFunction
  );

  //Training;
  await model.fitDataset(dividedData, {
    epochs: epochs,
    verbose: 0,
    callbacks: {
      onEpochEnd(epoch, logs?) {
        if (logs) {
          console.log(
            `Epoch: ${epoch}: loss: ${logs.loss.toFixed(
              4
            )},acc: ${logs.acc.toFixed(4)}`
          );
        }
      },
    },
  });

  // Validate model and create ErrorMatrix
  const validationResult = await validateModel(
    model,
    clearedData,
    numbOfClasses
  );
  createErrorMatrixFile(saveErrorMatrixFileName, validationResult);

  // Save created model
  await model.save(saveModelLocation);
}

main();
