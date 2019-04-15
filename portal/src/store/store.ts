import { configureStore } from "redux-starter-kit";
import { carsReducer } from "./cars/carsReducer";
import { RootState } from "./storeTypings";
import { tubsReducer } from "./tubs/tubsReducer";
import { uiReducer } from "./ui/uiReducer";
import { tubDataReducer } from "./tubData/tubDataReducer";

export const store = configureStore<RootState>({
  reducer: {
    cars: carsReducer,
    tubs: tubsReducer,
    tubData: tubDataReducer,
    ui: uiReducer
  }
});
