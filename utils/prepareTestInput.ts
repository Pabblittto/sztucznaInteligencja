import { BaseRowType } from "../types/baseTypes";
import { NormalizationType, getNormalizingFunction } from "./normalize";

export const prepareTestInput = async (
  data: BaseRowType,
  normalizationType: NormalizationType
) => {
  return await getNormalizingFunction(normalizationType)(data);
};
