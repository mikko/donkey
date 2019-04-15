import * as React from "react";
import { Row, Col } from "antd/lib/grid";
import { Link } from "react-router-dom";
import { CarCard } from "../CarCard/CarCard";

export interface Car {
  id: string;
  name: string;
}

export interface CarsGridProps {
  cars: Car[];
}

export const CarsGrid: React.SFC<CarsGridProps> = ({ cars }) => (
  <Row>
    {cars.map(car => (
      <Col
        key={car.id}
        className="landing-page-item"
        xs={24}
        sm={12}
        md={8}
        lg={8}
        xl={8}
        span={2}
      >
        <Link to={`/car/${car.id}`}>
          <CarCard name={car.name} />
        </Link>
      </Col>
    ))}
  </Row>
);
