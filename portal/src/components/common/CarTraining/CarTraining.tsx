import styles from "./CarTraining.module.css";
import * as React from "react";
import { Row, Col } from "antd/lib/grid";
import { CarTrainingParamsForm } from "../CarTrainingParamsForm/CarTrainingParamsForm";
import { Tub } from "../../../store/tubs/tubsTypings";

export interface CarTrainingProps {
  tubs: Tub[];
}

export const CarTraining: React.FunctionComponent<CarTrainingProps> = ({
  tubs
}) => {
  return (
    <Row className={styles.carTraining}>
      <Col>
        <CarTrainingParamsForm tubs={tubs} />
      </Col>
    </Row>
  );
};
