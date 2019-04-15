import styles from "./CarData.module.css";
import * as React from "react";
import Typography from "antd/lib/typography";
import Layout from "antd/lib/layout";
import { Row } from "antd/lib/grid";
import { TubSelector } from "../TubSelector/TubSelector";
import { Tub, TubDataPoint } from "../../../api";
import { TubStats } from "../TubStats/TubStats";
import TubPlayerLoading, { TubPlayer } from "../TubPlayer/TubPlayer";
const { Content, Sider } = Layout;
const { Title } = Typography;

export interface CarDataProps {
  tubs: Tub[];
  selectedTub?: Tub;
  selectedTubData?: ReadonlyArray<TubDataPoint>;
  onTubSelected: (tub: Tub) => void;
}

export const CarData: React.FunctionComponent<CarDataProps> = props => {
  const content = props.selectedTub ? (
    <div className={styles.tubDetails}>
      <Title>{props.selectedTub.name}</Title>
      <TubStats
        numDataPoints={props.selectedTub.numDataPoints}
        timestamp={props.selectedTub.timestamp}
      />
      <div className={styles.tubPlayerContainer}>
        {props.selectedTubData ? (
          <TubPlayer data={props.selectedTubData} />
        ) : (
          <TubPlayerLoading />
        )}
      </div>
    </div>
  ) : (
    "Please select a tub"
  );

  return (
    <Layout>
      <Sider>
        <TubSelector
          tubs={props.tubs}
          selectedTub={props.selectedTub}
          onTubSelected={tub => props.onTubSelected(tub)}
        />
      </Sider>
      <Layout>
        <Content>{content}</Content>
      </Layout>
    </Layout>
    // <Row gutter={2}>
    //   <Col sm={2} />
    //   <Col sm={4}>

    //   </Col>
    //   <Col sm={16}>
    //     {/* {props.selectedTub && props.selectedTubData ? (
    //       <TubDetails data={props.selectedTubData} tub={props.selectedTub} />
    //     ) : null} */}
    //   </Col>
    //   <Col sm={2} />
    // </Row>
  );
};
