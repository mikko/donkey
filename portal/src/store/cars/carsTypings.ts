export interface Car {
  id: string;
  name: string;
}

export interface CarsState {
  cars: {
    readonly [carId: string]: Car;
  };
  carTubs: {
    [carId: string]: string[];
  };
  isLoading: boolean;
}
