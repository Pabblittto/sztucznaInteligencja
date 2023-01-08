import {
  discMonth,
  discDay,
  discFFMC,
  discDMC,
  discDC,
  discISI,
  discTemp,
  discRH,
  discWind,
  discRain,
  discArea,
} from "./methods";
import { BaseRowType, DiscretizatedRowType } from "../../types/baseTypes";
import { TensorContainer } from "@tensorflow/tfjs";

/**
 * Function for normalizing rows
 * @param row Row
 * @returns normalized row
 */
export const expertNormalizeFn = async (row: TensorContainer) => {
  const typedRow: BaseRowType = row as BaseRowType;

  const result: DiscretizatedRowType = {
    X: typedRow.X,
    Y: typedRow.Y,
    month: await discMonth(typedRow.month),
    day: await discDay(typedRow.day),
    FFMC: await discFFMC(typedRow.FFMC),
    DMC: await discDMC(typedRow.DMC),
    DC: await discDC(typedRow.DC),
    ISI: await discISI(typedRow.ISI),
    temp: await discTemp(typedRow.temp),
    RH: await discRH(typedRow.RH),
    wind: await discWind(typedRow.wind),
    rain: await discRain(typedRow.rain),
    area: await discArea(typedRow.area),
  };

  return result;
};
