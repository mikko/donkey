import { createAction } from "typesafe-actions";
import { Tub } from "./tubsTypings";
import { Dispatch } from "redux";
import { tubApi } from "../../api";
import { loadCarById } from "../cars";
import { StoreDispatch } from "../storeTypings";

export const setTubs = createAction(
  "tubs/SET_TUBS",
  action => (carId: string, tubs: Tub[]) => action({ carId, tubs })
);

export const setIsLoadingTubs = createAction(
  "tubs/SET_IS_LOADING",
  action => (isLoading: boolean) => action({ isLoading })
);

export const loadTubs = (carId: string) => async (dispatch: StoreDispatch) => {
  dispatch(setIsLoadingTubs(true));

  await dispatch(loadCarById(carId));

  try {
    const foundTubs = await tubApi.getTubsByCar({ carId });

    dispatch(setTubs(carId, foundTubs));
  } catch (error) {
  } finally {
    dispatch(setIsLoadingTubs(false));
  }
};
