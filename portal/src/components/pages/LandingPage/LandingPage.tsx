import * as React from "react";
import "./LandingPage.css";
import Layout from "antd/lib/layout";
import Spin from "antd/lib/spin";
import { CarsGrid, Car } from "../../common/CarsGrid/CarsGrid";
import Alert from "antd/lib/alert";
import { connect } from "react-redux";
import { getCars, getIsLoadingCars, loadCars } from "../../../store/cars";
import { RootState } from "../../../store/storeTypings";
import { useDidMount } from "../../../utils/hooks/didMount";
import { store } from "../../../store/store";
const { Header, Footer, Content } = Layout;

interface LandingPageProps {
  isLoading: boolean;
  cars: Car[];
  error?: string;
}

const LandingPage: React.FunctionComponent<LandingPageProps> = ({
  cars,
  isLoading,
  error
}) => {
  useDidMount(() => {
    store.dispatch(loadCars());
  });

  const content = isLoading ? (
    <Spin style={{ margin: "3rem" }} size="large" />
  ) : error ? (
    <Alert message="Error loading cars" description={error} type="error" />
  ) : (
    <CarsGrid cars={cars} />
  );

  return (
    <Layout>
      <Header>Markku portal</Header>
      <Content className="landing-page">{content}</Content>
      <Footer>Footer</Footer>
    </Layout>
  );
};

const mapStateToProps = (state: RootState) => {
  return {
    cars: getCars(state),
    isLoading: getIsLoadingCars(state)
  };
};

const ConnectedLandingPage = connect(mapStateToProps)(LandingPage);

export { ConnectedLandingPage as LandingPage };
