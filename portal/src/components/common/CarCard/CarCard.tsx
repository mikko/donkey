import * as React from "react";
import carImage from "./markku_bg_blue.jpg";
import "./CarCard.module.css";
import Card from "antd/lib/card";

const { Meta } = Card;

export interface CarCardProps {
  name: string;
}

export const CarCard: React.FunctionComponent<CarCardProps> = ({ name }) => (
  <Card hoverable style={{ width: 240 }} cover={<img src={carImage} />}>
    <Meta title={name} description={name} />
  </Card>
);
