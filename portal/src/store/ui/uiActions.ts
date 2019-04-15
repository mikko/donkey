import { createAction } from "typesafe-actions";
import { getTubData } from "../tubData/tubDataSelectors";
import { loadTubDataPoints } from "../tubData/tubDataActions";
import { StoreDispatch, RootState } from "../storeTypings";

export const setSelectedCar = createAction(
  "ui/SELECT_CAR",
  action => (carId: string) => action({ carId })
);

export const setSelectedTub = createAction(
  "ui/SELECT_TUB",
  action => (tubId: string) => action({ tubId })
);

export const setSelectedDataPoint = createAction(
  "ui/SELECT_DATA_POINT",
  action => (dataPointId: number) => action({ dataPointId })
);

export const selectTub = (carId: string, tubId: string) => async (
  dispatch: StoreDispatch,
  getState: () => RootState
) => {
  dispatch(setSelectedTub(tubId));

  const tubDataPoints = getTubData(getState(), tubId);
  if (!tubDataPoints) {
    await dispatch(loadTubDataPoints(carId, tubId));
  }
};
