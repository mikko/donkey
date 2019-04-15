import { from } from "fromfrom";
import { RootState } from "../storeTypings";
import { getTubsById } from "../tubs/tubsSelectors";

export const getCars = (state: RootState) => Object.values(state.cars.cars);

export const getCarById = (state: RootState, carId: string) =>
  state.cars.cars[carId];

export const getCarTubs = (state: RootState, carId: string) => {
  const carTubIds = state.cars.carTubs[carId] || [];

  return from(carTubIds)
    .map(tubId => getTubsById(state, tubId))
    .toArray();
};

export const getIsLoadingCars = (state: RootState) => state.cars.isLoading;
