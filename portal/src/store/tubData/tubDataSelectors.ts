import { RootState } from "../storeTypings";

export const isLoadingForTub = (state: RootState, tubId: string) =>
  state.tubData.isLoadingForTubs.includes(tubId);

export const getTubData = (state: RootState, tubId: string) =>
  state.tubData.tubDataPoints[tubId];

export const getTubDataPointById = (
  state: RootState,
  tubId: string,
  dataPointId: number
) => {
  const tubDataPoints = state.tubData.tubDataPoints[tubId];
  if (!tubDataPoints) {
    return undefined;
  }

  return tubDataPoints[dataPointId];
};
