import styles from "./CarPage.module.css";
import * as React from "react";
// import { Row, Col } from "antd/lib/grid";
import Layout from "antd/lib/layout";
import Spin from "antd/lib/spin";
import { Navigation, CarDataPage } from "../../common/Navigation/Navigation";
import { CarData } from "../../common/CarData/CarData";
import { Car, getCarById, getCarTubs } from "../../../store/cars";
import { Tub } from "../../../store/tubs/tubsTypings";
import { useDidMount } from "../../../utils/hooks/didMount";
import { loadTubs } from "../../../store/tubs/tubsActions";
import { RootState, StoreDispatch } from "../../../store/storeTypings";
import { connect } from "react-redux";
import {
  getIsLoadingTubs,
  getTubsById
} from "../../../store/tubs/tubsSelectors";
import { store } from "../../../store/store";
import { getSelectedTubId } from "../../../store/ui/uiSelectors";
import { selectTub } from "../../../store/ui/uiActions";
import { TubDataPoint } from "../../../store/tubData/tubDataTypings";
import { getTubData } from "../../../store/tubData/tubDataSelectors";
import { CarTraining } from "../../common/CarTraining/CarTraining";

const { Header, Footer, Content } = Layout;

export interface CarPageOwnProps {
  carId: string;
}

export interface CarPageProps {
  isLoading: boolean;
  car: Car;
  tubs: Tub[];
  selectedTub?: Tub;
  selectedTubData?: ReadonlyArray<TubDataPoint>;
  selectedPage?: CarDataPage;
}

export interface CarPageDispatchProps {
  tubSelected: (tub: Tub) => void;
}

const CarPage: React.FunctionComponent<
  CarPageProps & CarPageOwnProps & CarPageDispatchProps
> = props => {
  const { selectedPage = CarDataPage.Data } = props;

  useDidMount(() => {
    store.dispatch(loadTubs(props.carId));
  });

  let content;
  if (props.isLoading) {
    content = <Spin style={{ margin: "3rem" }} size="large" />;
  } else if (!props.selectedPage || props.selectedPage === CarDataPage.Data) {
    content = (
      <CarData
        onTubSelected={props.tubSelected}
        selectedTub={props.selectedTub}
        tubs={props.tubs}
        selectedTubData={props.selectedTubData}
      />
    );
  } else if (props.selectedPage === CarDataPage.Train) {
    content = <CarTraining tubs={props.tubs} />;
  }

  return (
    <Layout>
      <Header className={styles.header}>
        <div className={styles.selectedCar}>
          {props.car ? props.car.name : ""}
        </div>
        <Navigation selectedCarId={props.carId} selectedPage={selectedPage} />
      </Header>

      <Content>{content}</Content>
      <Footer>Footer</Footer>
    </Layout>
  );
};

const mapStateToProps = (state: RootState, ownProps: CarPageOwnProps) => {
  const selectedTubId = getSelectedTubId(state);

  return {
    car: getCarById(state, ownProps.carId),
    tubs: getCarTubs(state, ownProps.carId),
    isLoading: getIsLoadingTubs(state),
    selectedTub: selectedTubId ? getTubsById(state, selectedTubId) : undefined,
    selectedTubData: selectedTubId
      ? getTubData(state, selectedTubId)
      : undefined
  };
};

const mapDispatchToProps = (
  dispatch: StoreDispatch,
  ownProps: CarPageOwnProps
) => {
  return {
    tubSelected: (tub: Tub) => dispatch(selectTub(ownProps.carId, tub.id))
  };
};

const ConnectedCarPage = connect(
  mapStateToProps,
  mapDispatchToProps
)(CarPage);

export { ConnectedCarPage as CarPage };
