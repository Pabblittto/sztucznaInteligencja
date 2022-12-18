import { correlatedRowRemover } from "../normalization/corelatedRowRemover";
import { BaseRowType } from "../types/baseTypes";
import { NormalizationType, getNormalizingFunction } from "./normalize";

export const prepareTestInput = async (
  data: BaseRowType,
  normalizationType: NormalizationType
) => {
  return correlatedRowRemover(
    await getNormalizingFunction(normalizationType)(data)
  );
};
