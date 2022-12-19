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
import { saveResults } from "./utils/saveBestResults";
const hpjs = require("hyperparameters");

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

  const saveErrorMatrixFileName = "errorMatrix"; // File name of error matrix file

  /**
   * Normalize type - define how data should be normalized
   */
  const normalizeType: NormalizationType = NormalizationType.EXPERT;

  const numbOfClasses = 5; // 5 number of calsses because ther is 5 types of area. See:  AreaThresholds

  const batchSize: number | undefined = undefined; // Batch size, set to undefined if you want to use default batch size
  const epochs = 10; // Number of learning times
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

  // TODO: DODAĆ KNN klasyfikator
  // Save clear data to file
  await saveDatasetAsCsv(`cleared_${normalizeType}`, finalDataset);

  const dividedData = await finalDataset
    .mapAsync((x) => datasetDivider(x, numbOfClasses))
    .toArray();

  const xs = dividedData.map((d) => d.xs);
  const ys = dividedData.map((d) => d.ys);

  // All possible elements that can be changed
  const space = {
    optimizer: hpjs.choice([
      Optimizer.adagrad,
      Optimizer.adam,
      Optimizer.rmsprop,
      Optimizer.sgd,
    ]),
    epochs: epochs,
    lossFunction: hpjs.choice([
      LossFunction.categoricalCrossentropy,
      LossFunction.meanSquaredError,
    ]),
    numbOfInternalLayers: hpjs.choice([1, 2, 3, 4]),
    activationFn: hpjs.choice(["relu", "sigmoid", "tanh"]),
  };

  //GRID SEARCH STEP. Create and train model:
  const createAndTrainModel = async (
    args: typeof space,
    { xs, ys }: { xs: tf.Tensor<tf.Rank.R1>[]; ys: tf.Tensor<tf.Rank.R1>[] }
  ) => {
    const model = await createNeuralNetworkModel(
      numbOfClasses,
      finalBatchSize,
      args.optimizer,
      args.lossFunction,
      args.numbOfInternalLayers,
      args.activationFn
    );

    const h = await model.fit(tf.stack(xs), tf.stack(ys), {
      epochs: args.epochs,
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

    const validationResult = await validateModel(
      model,
      clearedData,
      numbOfClasses
    );

    const lastLoss = h.history.loss[h.history.loss.length - 1];

    createErrorMatrixFile(
      `${saveErrorMatrixFileName}_${args.optimizer}_${args.lossFunction}_layers${args.numbOfInternalLayers}_${args.activationFn}_loss${lastLoss}.html`,
      validationResult
    );

    return { model, loss: lastLoss };
  };

  const modelOpt = async (
    args: typeof space,
    { xs, ys }: { xs: tf.Tensor<tf.Rank.R1>[]; ys: tf.Tensor<tf.Rank.R1>[] }
  ) => {
    const { loss } = await createAndTrainModel(args, { xs, ys });
    return { loss, status: hpjs.STATUS_OK };
  };

  const trials = await hpjs.fmin(
    modelOpt,
    space,
    hpjs.search.randomSearch,
    10,
    { rng: new hpjs.RandomState(654321), xs, ys }
  );
  const opt = trials.argmin;
  console.log(opt);
  saveResults(opt, trials);
}

main();
