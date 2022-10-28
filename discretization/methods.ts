import {
  monthMap,
  dayMap,
  FFMCthresholds,
  DMCthresholds,
  DCthresholds,
  ISIthresholds,
  Tempthresholds,
  RHthresholds,
  Windthresholds,
  Rainthresholds,
  AreaThresholds,
} from "./values";
import * as tf from "@tensorflow/tfjs";
// File that contains helper methods for discretization

import { Month, Day } from "../types/baseTypes";
/**
 * WorkAround
 */
const tensorToNumber = (tensor: tf.Tensor) => {
  return Number(tensor.arraySync().toString());
};

export const discMonth = async (month: Month) => {
  return monthMap(month);
};

export const discDay = async (day: Day) => {
  return dayMap(day);
};

export const discFFMC = async (FFMC: number) => {
  const FFMCValue = tf.tensor1d([FFMC]);
  const res = tf.upperBound(FFMCthresholds, FFMCValue);

  return tensorToNumber(res);
};

export const discDMC = async (DMC: number) => {
  const DMCValue = tf.tensor1d([DMC]);
  const res = tf.upperBound(DMCthresholds, DMCValue);

  return tensorToNumber(res);
};

export const discDC = async (DC: number) => {
  const DCValue = tf.tensor1d([DC]);
  const res = tf.upperBound(DCthresholds, DCValue);

  return tensorToNumber(res);
};
export const discISI = async (ISI: number) => {
  const ISIValue = tf.tensor1d([ISI]);
  const res = tf.upperBound(ISIthresholds, ISIValue);

  return tensorToNumber(res);
};

export const discTemp = async (temp: number) => {
  const tempValue = tf.tensor1d([temp]);
  const res = tf.upperBound(Tempthresholds, tempValue);

  return tensorToNumber(res);
};
export const discRH = async (RH: number) => {
  const RHValue = tf.tensor1d([RH]);
  const res = tf.upperBound(RHthresholds, RHValue);

  return tensorToNumber(res);
};
export const discWind = async (wind: number) => {
  const WindValue = tf.tensor1d([wind]);
  const res = tf.upperBound(Windthresholds, WindValue);

  return tensorToNumber(res);
};
export const discRain = async (rain: number) => {
  const RainValue = tf.tensor1d([rain]);
  const res = tf.lowerBound(Rainthresholds, RainValue);

  return tensorToNumber(res);
};

export const discArea = async (area: number) => {
  const AreaValue = tf.tensor1d([area]);
  const res = tf.lowerBound(AreaThresholds, AreaValue);

  return tensorToNumber(res);
};
