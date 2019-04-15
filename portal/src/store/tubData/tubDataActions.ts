import { createAction } from "typesafe-actions";
import { TubDataPoint } from "./tubDataTypings";
import { tubDataApi } from "../../api";
import { StoreDispatch, RootState } from "../storeTypings";
import { getTubDataPointById, isLoadingForTub } from "./tubDataSelectors";

export const setTubDataPoints = createAction(
  "tubData/SET_TUB_DATA",
  action => (tubId: string, tubDataPoints: TubDataPoint[]) =>
    action({ tubId, tubDataPoints })
);

export const setLoadingTubData = createAction(
  "tubData/SET_IS_LOADING",
  action => (tubId: string) => action({ tubId })
);

export const setFinishLoadingTubData = createAction(
  "tubData/FINISH_LOADING",
  action => (tubId: string) => action({ tubId })
);

/**
 * Load all data points for given tub. No-op if already loading
 */
export const loadTubDataPoints = (carId: string, tubId: string) => async (
  dispatch: StoreDispatch,
  getState: () => RootState
) => {
  if (isLoadingForTub(getState(), tubId)) {
    return;
  }
  dispatch(setLoadingTubData(tubId));

  try {
    const dataPoints = await tubDataApi.getTubDataPoints({ carId, tubId });

    dispatch(setTubDataPoints(tubId, dataPoints));
  } catch (error) {
  } finally {
    dispatch(setFinishLoadingTubData(tubId));
  }
};

export const setTubDataPoint = createAction(
  "tubDataPoints/SET_TUBS",
  action => (tubId: string, dataPointId: number, tubDataPoint: TubDataPoint) =>
    action({ tubId, dataPointId, tubDataPoint })
);

export const loadDataPointIfNeeded = (
  carId: string,
  tubId: string,
  dataPointId: number
) => async (dispatch: StoreDispatch, getState: () => RootState) => {
  if (getTubDataPointById(getState(), tubId, dataPointId)) {
    return;
  }

  try {
    const dataPoint = await tubDataApi.getTubDataPointByCarAndId({
      carId,
      tubId,
      dataId: dataPointId.toString()
    });

    dispatch(setTubDataPoint(tubId, dataPointId, dataPoint));
  } catch (error) {}
};
