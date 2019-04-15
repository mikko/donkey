import React from "react";
import "../App.css";

import { storiesOf } from "@storybook/react";
// import { action } from "@storybook/addon-actions";
// import { linkTo } from "@storybook/addon-links";

import { TubSelector } from "../components/common/TubSelector/TubSelector";
import TubPlayerLoading, {
  TubPlayer
} from "../components/common/TubPlayer/TubPlayer";
import { TubDetails } from "../components/common/TubDetails/TubDetails";
import { CarsGrid } from "../components/common/CarsGrid/CarsGrid";
import { CarTrainingParamsForm } from "../components/common/CarTrainingParamsForm/CarTrainingParamsForm";

// storiesOf("Welcome", module).add("to Storybook", () => (
//   <Welcome showApp={linkTo("Button")} />
// ));

// storiesOf("Button", module)
//   .add("with text", () => (
//     <Button onClick={action("clicked")}>Hello Button</Button>
//   ))
//   .add("with some emoji", () => (
//     <Button onClick={action("clicked")}>
//       <span role="img" aria-label="so cool">
//         😀 😎 👍 💯
//       </span>
//     </Button>
//   ));

storiesOf("CarsGrid", module).add("with some cars", () => (
  <div style={{ margin: "2rem" }}>
    <CarsGrid
      cars={[
        { id: "1", name: "Markku" },
        { id: "2", name: "Mirkku" },
        { id: "3", name: "Perttu" },
        { id: "4", name: "Martti" },
        { id: "5", name: "Salama McQueen" }
      ]}
    />
  </div>
));

storiesOf("TubSelector", module).add("with two tubs", () => (
  <div style={{ margin: "2rem" }}>
    <TubSelector
      onTubSelected={() => {}}
      tubs={[
        {
          id: "Tub 1",
          name: "Tub 1",
          timestamp: new Date(2019, 3, 3, 15, 0),
          numDataPoints: 20
        },
        {
          id: "Tub 2",
          name: "Tub 2",
          timestamp: new Date(2019, 3, 3, 15, 10),
          numDataPoints: 20
        }
      ]}
    />
  </div>
));

storiesOf("TubPlayer", module).add("with some data", () => (
  <div style={{ margin: "2rem 4rem" }}>
    <TubPlayer
      // prettier-ignore
      data={[
        { angle: 1,  timestamp: new Date(2019, 3, 3, 10, 0), throttle: 1, imageName: "example.jpg" },
        { angle: 2,  timestamp: new Date(2019, 3, 3, 10, 0), throttle: 2, imageName: "example.jpg" },
        { angle: 3,  timestamp: new Date(2019, 3, 3, 10, 0), throttle: 3, imageName: "example.jpg" },
        { angle: 4,  timestamp: new Date(2019, 3, 3, 10, 0), throttle: 4, imageName: "example.jpg" },
        { angle: 5,  timestamp: new Date(2019, 3, 3, 10, 0), throttle: 5, imageName: "example.jpg" },
        { angle: 6,  timestamp: new Date(2019, 3, 3, 10, 0), throttle: 6, imageName: "example.jpg" },
        { angle: 7,  timestamp: new Date(2019, 3, 3, 10, 0), throttle: 7, imageName: "example.jpg" },
        { angle: 8,  timestamp: new Date(2019, 3, 3, 10, 0), throttle: 8, imageName: "example.jpg" },
        { angle: 9,  timestamp: new Date(2019, 3, 3, 10, 0), throttle: 9, imageName: "example.jpg" },
        { angle: 10, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 10, imageName: "example.jpg" },
        { angle: 11, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 11, imageName: "example.jpg" },
        { angle: 12, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 12, imageName: "example.jpg" },
        { angle: 13, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 13, imageName: "example.jpg" },
        { angle: 14, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 14, imageName: "example.jpg" },
        { angle: 15, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 15, imageName: "example.jpg" },
        { angle: 16, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 16, imageName: "example.jpg" },
        { angle: 17, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 17, imageName: "example.jpg" },
        { angle: 18, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 18, imageName: "example.jpg" },
        { angle: 19, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 19, imageName: "example.jpg" },
        { angle: 20, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 20, imageName: "example.jpg" }
      ]}
    />
  </div>
));

storiesOf("TubPlayer", module)
  .add("with some data", () => (
    <div style={{ margin: "2rem 4rem" }}>
      <TubPlayer
        // prettier-ignore
        data={[
        { angle: 1,  timestamp: new Date(2019, 3, 3, 10, 0), throttle: 1, imageName: "example.jpg" },
        { angle: 2,  timestamp: new Date(2019, 3, 3, 10, 0), throttle: 2, imageName: "example.jpg" },
        { angle: 3,  timestamp: new Date(2019, 3, 3, 10, 0), throttle: 3, imageName: "example.jpg" },
        { angle: 4,  timestamp: new Date(2019, 3, 3, 10, 0), throttle: 4, imageName: "example.jpg" },
        { angle: 5,  timestamp: new Date(2019, 3, 3, 10, 0), throttle: 5, imageName: "example.jpg" },
        { angle: 6,  timestamp: new Date(2019, 3, 3, 10, 0), throttle: 6, imageName: "example.jpg" },
        { angle: 7,  timestamp: new Date(2019, 3, 3, 10, 0), throttle: 7, imageName: "example.jpg" },
        { angle: 8,  timestamp: new Date(2019, 3, 3, 10, 0), throttle: 8, imageName: "example.jpg" },
        { angle: 9,  timestamp: new Date(2019, 3, 3, 10, 0), throttle: 9, imageName: "example.jpg" },
        { angle: 10, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 10, imageName: "example.jpg" },
        { angle: 11, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 11, imageName: "example.jpg" },
        { angle: 12, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 12, imageName: "example.jpg" },
        { angle: 13, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 13, imageName: "example.jpg" },
        { angle: 14, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 14, imageName: "example.jpg" },
        { angle: 15, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 15, imageName: "example.jpg" },
        { angle: 16, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 16, imageName: "example.jpg" },
        { angle: 17, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 17, imageName: "example.jpg" },
        { angle: 18, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 18, imageName: "example.jpg" },
        { angle: 19, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 19, imageName: "example.jpg" },
        { angle: 20, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 20, imageName: "example.jpg" }
      ]}
      />
    </div>
  ))
  .add("loading", () => (
    <div style={{ margin: "2rem 4rem" }}>
      <TubPlayerLoading />
    </div>
  ));

storiesOf("CarTrainingParams", module).add("empty form", () => (
  <div style={{ margin: "2rem 4rem" }}>
    <CarTrainingParamsForm
      tubs={[
        {
          id: "Tub 1",
          name: "Tub 1",
          timestamp: new Date(2019, 3, 3, 15, 10),
          numDataPoints: 3210
        },
        {
          id: "Tub 2",
          name: "Tub 2",
          timestamp: new Date(2019, 3, 3, 15, 10),
          numDataPoints: 4223
        }
      ]}
    />
  </div>
));

// storiesOf("TubDetails", module).add("with some data", () => (
//   <div style={{ margin: "2rem 4rem" }}>
//     <TubDetails
//       tub={{
//         id: "Tub 1",
//         name: "Tub 1",
//         timestamp: new Date(2019, 3, 3, 15, 0),
//         numDataPoints: 20
//       }}
//       data={
//         // prettier-ignore
//         [
//           { angle: 1,  timestamp: new Date(2019, 3, 3, 10, 0), throttle: 1, imageName: "example.jpg" },
//           { angle: 2,  timestamp: new Date(2019, 3, 3, 10, 0), throttle: 2, imageName: "example.jpg" },
//           { angle: 3,  timestamp: new Date(2019, 3, 3, 10, 0), throttle: 3, imageName: "example.jpg" },
//           { angle: 4,  timestamp: new Date(2019, 3, 3, 10, 0), throttle: 4, imageName: "example.jpg" },
//           { angle: 5,  timestamp: new Date(2019, 3, 3, 10, 0), throttle: 5, imageName: "example.jpg" },
//           { angle: 6,  timestamp: new Date(2019, 3, 3, 10, 0), throttle: 6, imageName: "example.jpg" },
//           { angle: 7,  timestamp: new Date(2019, 3, 3, 10, 0), throttle: 7, imageName: "example.jpg" },
//           { angle: 8,  timestamp: new Date(2019, 3, 3, 10, 0), throttle: 8, imageName: "example.jpg" },
//           { angle: 9,  timestamp: new Date(2019, 3, 3, 10, 0), throttle: 9, imageName: "example.jpg" },
//           { angle: 10, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 10, imageName: "example.jpg" },
//           { angle: 11, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 11, imageName: "example.jpg" },
//           { angle: 12, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 12, imageName: "example.jpg" },
//           { angle: 13, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 13, imageName: "example.jpg" },
//           { angle: 14, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 14, imageName: "example.jpg" },
//           { angle: 15, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 15, imageName: "example.jpg" },
//           { angle: 16, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 16, imageName: "example.jpg" },
//           { angle: 17, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 17, imageName: "example.jpg" },
//           { angle: 18, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 18, imageName: "example.jpg" },
//           { angle: 19, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 19, imageName: "example.jpg" },
//           { angle: 20, timestamp: new Date(2019, 3, 3, 10, 0), throttle: 20, imageName: "example.jpg" }
//         ]
//       }
//     />
//   </div>
// ));
