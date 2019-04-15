import { createAction } from "typesafe-actions";
import { Car } from "./carsTypings";
import { Dispatch, AnyAction } from "redux";
import { carApi } from "../../api";
import { getCarById } from "./carsSelectors";
import { RootState, StoreDispatch } from "../storeTypings";
import { ThunkDispatch } from "redux-thunk";

export const setCars = createAction("cars/SET_CARS", action => (cars: Car[]) =>
  action({ cars })
);

export const setIsLoadingCars = createAction(
  "cars/SET_IS_LOADING",
  action => (isLoading: boolean) => action({ isLoading })
);

export const loadCarById = (carId: string) => async (
  dispatch: StoreDispatch,
  getState: () => RootState
) => {
  const state = getState();

  const car = getCarById(state, carId);
  if (!car) {
    await dispatch(loadCars());
  }
};

export const loadCars = () => async (dispatch: StoreDispatch) => {
  dispatch(setIsLoadingCars(true));

  try {
    const foundCars = await carApi.getCars();

    dispatch(setCars(foundCars));
  } catch (error) {
    console.error(error);
  } finally {
    dispatch(setIsLoadingCars(false));
  }
};
