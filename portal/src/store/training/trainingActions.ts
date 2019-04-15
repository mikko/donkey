import { createAction } from "typesafe-actions";
import { TrainingStatus } from "./trainingTypings";
import { StoreDispatch } from "../storeTypings";
import { trainingApi } from "../../api";

export const setTrainingStatus = createAction(
  "training/SET_TRAINING_STATUS",
  action => (status: TrainingStatus) => action({ status })
);

export const setIsLoadingTrainingStatus = createAction(
  "training/SET_IS_LOADING",
  action => (isLoading: boolean) => action({ isLoading })
);

export const loadTrainingStatus = (carId: string) => async (
  dispatch: StoreDispatch
) => {
  dispatch(setIsLoadingTrainingStatus(true));

  try {
    const trainingStatus = await trainingApi.getTrainingInfoByCarId({ carId });

    dispatch(setTrainingStatus(trainingStatus));
  } catch (error) {
    console.error(error);
  } finally {
    dispatch(setIsLoadingTrainingStatus(false));
  }
};
