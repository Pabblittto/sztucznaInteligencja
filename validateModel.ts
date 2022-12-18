import * as tf from "@tensorflow/tfjs";
import "tfjs-node-save"; // very important line!
import { prepareTestInput } from "./utils/prepareTestInput";
import { BaseRowType, ClearedRowData, Day, Month } from "./types/baseTypes";
import { datasetDivider } from "./utils/datasetDivider";
import { NormalizationType } from "./utils/normalize";

const main = async () => {
  const data: BaseRowType = {
    X: 3, // pozycja X w parku
    Y: 5, // pozycja Y w parku
    day: Day.mon, // dzień tygodnia
    month: Month.jul, // miesiąc
    DC: 1, // wilgotność głębokich, zwartych elemntów ściółki (0-1000)
    DMC: 1, // wilgotność średnio zbitych elemntów ściółki (0-100)
    FFMC: 1, // wilgotność drobnych, łatwopalnych elemmentów ściółki (0-100)
    RH: 1, // relatywna wilgotność (0%-100%)
    temp: 33, // in celcjus degrees
    rain: 0, // in mm/m^2
    wind: 7, // prędkość w km/h
    ISI: 70, // wartość liczona na podstawie FFMC i prędkości wiatru, określa szybkość rozprzestrzeniania się ognia (0-60) około
    area: 0, // Zostawić puste, niezmieniać, dana jest pomijana i nieprzekazywana do modelu
  };

  /**
   * Must be the same like the one used during creating a model
   */
  const normalizationType: NormalizationType = NormalizationType.EXPERT;
  const numberOfClasses = 5; // number of final classes, must be the same like the one during creating a model

  const modelLocation =
    process.platform === "win32"
      ? "file://" + __dirname + "\\savedModels\\model.json"
      : "file://" + __dirname + "/savedModels/model.json";

  console.log(modelLocation);

  const model = await tf.loadLayersModel(modelLocation);

  const preparedData = await prepareTestInput(data, normalizationType);

  const asd: ClearedRowData = {
    X: 4,
    Y: 5,
    day: 6,
    temp: 1,
    RH: 2,
    wind: 2,
    rain: 0,
    area: 4,
  };

  const testTensor = await datasetDivider(asd, numberOfClasses);

  const result = model.predict(testTensor.xs.reshape([1, 7]));
  console.log(result.toString());
};

main();
