import { createReducer } from "redux-starter-kit";
import { TrainingState } from "./trainingTypings";
import { getType } from "typesafe-actions";
import {
  setTrainingStatus,
  setIsLoadingTrainingStatus
} from "./trainingActions";

const initialState: TrainingState = {
  isLoading: false,
  params: {},
  status: {
    status: "not_training"
  }
} as const;

export const trainingReducer = createReducer<TrainingState, any>(initialState, {
  [getType(setTrainingStatus)]: (
    state,
    action: ReturnType<typeof setTrainingStatus>
  ) => {
    state.status = action.payload.status;
  },

  [getType(setIsLoadingTrainingStatus)]: (
    state,
    action: ReturnType<typeof setIsLoadingTrainingStatus>
  ) => {
    state.isLoading = action.payload.isLoading;
  }
});
