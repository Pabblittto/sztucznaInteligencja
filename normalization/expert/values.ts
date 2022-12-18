import * as tf from "@tensorflow/tfjs";
import { Day, Month } from "../../types/baseTypes";

// FYI: need to use upperBound to make it work

/**
 * UpperBound method results means following discretizated values:
 *
 *  (character '<' means the number is included in that particular rating)
 *
 * - 1 - Low <0-76)
 * - 2 - Moderate <76-84)
 * - 3 - High <84-88)
 * - 4 - Very High <88-91)
 * - 5 - Extreme <91-inf)
 *
 */
export const FFMCthresholds = tf.tensor1d([0, 76, 84, 88, 91]);

/**
 * UpperBound method results means following discretizated values:
 *  (character '<' means the number is included in that particular rating)
 *
 * - 1 - Low <0-21)
 * - 2 - Moderate <21-27)
 * - 3 - High <27-40)
 * - 4 - Very High <40-60)
 * - 5 - Extreme <60-inf)
 *
 */
export const DMCthresholds = tf.tensor1d([0, 21, 27, 40, 60]);

/**
 * UpperBound method results means following discretizated values:
 *  (character '<' means the number is included in that particular rating)
 *
 * - 1 - Low <0-79)
 * - 2 - Moderate <79-189)
 * - 3 - High <189-299)
 * - 4 - Very High <299-424)
 * - 5 - Extreme <424-inf)
 *
 */
export const DCthresholds = tf.tensor1d([0, 79, 189, 299, 424]);

/**
 * UpperBound method results means following discretizated values:
 *  (character '<' means the number is included in that particular rating)
 *
 * - 1 - Low <0-1.5)
 * - 2 - Moderate <1.5-4.0)
 * - 3 - High <4.0-8.0)
 * - 4 - Very High <8.0-15)
 * - 5 - Extreme <15-inf)
 *
 */
export const ISIthresholds = tf.tensor1d([0, 1.5, 4.0, 8.0, 15]);

/**
 * UpperBound method results means following discretizated values:
 *  (character '<' means the number is included in that particular rating)
 *
 * - 0 - Cold <-inf - 11.5 )
 * - 1 - Moderate <11.5-19)
 * - 2 - Warm <19-26)
 * - 3 - Hot <26-inf)
 *
 */
export const Tempthresholds = tf.tensor1d([11.5, 19, 26]);

/**
 * UpperBound method results means following discretizated values:
 *  (character '<' means the number is included in that particular rating)
 *
 * - 1 - <0-20)
 * - 2 - <20-40)
 * - 3 - <40-60)
 * - 4 - <60-80)
 * - 5 - <80-100)
 */
export const RHthresholds = tf.tensor1d([0, 20, 40, 60, 80]);

/**
 * UpperBound method results means following discretizated values:
 *  (character '<' means the number is included in that particular rating)
 *  Wind speed is divided based on Beaufort wind scale
 *
 * - 1 - Calm <0-1)
 * - 2 - Light Air <1-5)
 * - 3 - Light Breeze <5-11)
 *
 */
export const Windthresholds = tf.tensor1d([0, 1, 5, 11]);

/**
 * BottomBond method results means following discretizated values:
 *  (character '<' means the number is included in that particular rating)
 *
 *(-inf - 0> – 0 (No rain)
 *(0 – 1> – 1 (Little rain)
 *(1 – 3> – 2 (Normal rain)
 *(3 –6> – 3 (Heavy rain)
 *
 */
export const Rainthresholds = tf.tensor1d([0, 1, 3, 6]);

/**
 * BottomBond method results means following discretizated values:
 *  (character '<' means the number is included in that particular rating)
 *
 *(-inf - 0> – 0 (no area)
 *(0 - 5> – 1 (small area)
 *(5 - 20> – 2 (medium area)
 *(20 - 100> – 3 (big area)
 *(100 - inf> – 4 (catastrofic big area)
 *
 *
 */
export const AreaThresholds = tf.tensor1d([0, 5, 20, 100]);

/**
 * Changes index of area type to readable label
 * @param index
 */
export const fromAreaIndexToLabel = (index: number): string => {
  switch (index) {
    case 0:
      return "no area";
    case 1:
      return "small area";
    case 2:
      return "medium area";
    case 3:
      return "big area";
    case 4:
      return "catastrofic area";

    default:
      throw "Should never happen";
  }
};

/**
 * Maps day to proper number
 * @param day
 * @returns
 */
export const dayMap = (day: Day): number => {
  switch (day) {
    case Day.mon:
      return 1;
    case Day.tue:
      return 2;
    case Day.wed:
      return 3;
    case Day.thu:
      return 4;
    case Day.fri:
      return 5;
    case Day.sat:
      return 6;
    case Day.sun:
      return 7;
    default:
      throw "Unsupported day: " + day;
  }
};

/**
 * Maps month to proper number
 * @param month
 * @returns
 */
export const monthMap = (month: Month): number => {
  switch (month) {
    case Month.jan:
      return 1;
    case Month.feb:
      return 2;
    case Month.mar:
      return 3;
    case Month.apr:
      return 4;
    case Month.may:
      return 5;
    case Month.jun:
      return 6;
    case Month.jul:
      return 7;
    case Month.aug:
      return 8;
    case Month.sep:
      return 9;
    case Month.oct:
      return 10;
    case Month.nov:
      return 11;
    case Month.dec:
      return 12;
    default:
      throw "unsupported month: " + month;
  }
};
