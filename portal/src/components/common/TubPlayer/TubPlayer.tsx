import styles from "./TubPlayer.module.css";
import React, { useEffect, useRef, useState } from "react";
import Typography from "antd/lib/typography";
import Slider, { SliderValue } from "antd/lib/slider";
import Icon from "antd/lib/icon";
import Spin from "antd/lib/spin";
import { TubDataPoint } from "../../../api";

const { Text } = Typography;

export interface TubPlayerProps {
  data: ReadonlyArray<TubDataPoint>;
}

export const TubPlayer: React.FunctionComponent<TubPlayerProps> = ({
  data
}) => {
  const [selectedDataPointIdx, setDataPointIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  if (selectedDataPointIdx > data.length) {
    setDataPointIdx(0);
  }
  const selectedDataPoint = data[selectedDataPointIdx];

  const onSelectedDataPointChanged = (newValue: SliderValue) =>
    typeof newValue === "number" ? setDataPointIdx(newValue - 1) : null;

  const onTogglePlaying = () => setIsPlaying(!isPlaying);

  useInterval(() => {
    if (isPlaying) {
      setDataPointIdx((selectedDataPointIdx + 1) % data.length);
    }
  }, 100);

  return (
    <div className={styles.tubPlayer}>
      <img
        className={styles.tubPlayerImage}
        src={selectedDataPoint && selectedDataPoint.imageName}
      />

      <div className={styles.tubplayerslidercontrol}>
        <Slider
          className={styles.tubplayerslider}
          min={1}
          max={data.length}
          value={selectedDataPointIdx + 1}
          onChange={onSelectedDataPointChanged}
        />
        <div className={styles.tubplayerplaybutton}>
          {isPlaying ? (
            <Icon type="pause" onClick={onTogglePlaying} />
          ) : (
            <Icon type="caret-right" onClick={onTogglePlaying} />
          )}
        </div>
      </div>

      {/* <Row>
        <Col xs={12}>
          <Statistic title="Angle" value={selectedDataPoint.angle} />
        </Col>
        <Col xs={12}>
          <Statistic title="Throttle" value={selectedDataPoint.throttle} />
        </Col>
      </Row> */}

      <div className={styles.tubPlayerDetails}>
        <div className={styles.tubPlayerDataPair}>
          <Text className={styles.tubPlayerDataPairLabel}>Angle:</Text>
          <Text>{selectedDataPoint && selectedDataPoint.angle}</Text>
        </div>

        <div className={styles.tubPlayerDataPair}>
          <Text className={styles.tubPlayerDataPairLabel}>Throttle:</Text>
          <Text>{selectedDataPoint && selectedDataPoint.throttle}</Text>
        </div>
      </div>
    </div>
  );
};

export interface TubPlayerLoadingProps {}

/**
 * A mock version of TubPlayer that can be shown when data is still loading
 */
const TubPlayerLoading: React.FunctionComponent<TubPlayerLoadingProps> = () => {
  return (
    <div className={styles.tubPlayer}>
      <div className={styles.tubPlayerLoadingIndicator}>
        <Spin size="large" />
      </div>

      <div className={styles.tubplayerslidercontrol}>
        <Slider
          className={styles.tubplayerslider}
          disabled
          min={1}
          max={1}
          value={1}
        />
        <div className={styles.tubplayerplaybutton}>
          <Icon type="caret-right" />
        </div>
      </div>

      <div className={styles.tubPlayerDetails}>
        <div className={styles.tubPlayerDataPair}>
          <Text className={styles.tubPlayerDataPairLabel}>Angle:</Text>
          <Text />
        </div>

        <div className={styles.tubPlayerDataPair}>
          <Text className={styles.tubPlayerDataPairLabel}>Throttle:</Text>
          <Text />
        </div>
      </div>
    </div>
  );
};

export default TubPlayerLoading;

/**
 * From: https://overreacted.io/making-setinterval-declarative-with-react-hooks/#just-show-me-the-code
 */
function useInterval(callback: () => void, delay: number) {
  const savedCallback = useRef<() => void>();

  // Remember the latest callback.
  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  // Set up the interval.
  useEffect(() => {
    function tick() {
      savedCallback.current!();
    }

    if (delay !== null) {
      let id = setInterval(tick, delay);
      return () => clearInterval(id);
    }
  }, [delay]);
}

// import React, { useEffect, useRef, useState } from "react";
// import styles from "./TubPlayer.module.css";
// import Typography from "antd/lib/typography";
// // import { Row, Col } from "antd/lib/grid";
// // import Statistic from "antd/lib/statistic";
// import Slider, { SliderValue } from "antd/lib/slider";
// import Icon from "antd/lib/icon";
// import { TubDataPoint } from "../../data/apiClient";

// const { Text, Title } = Typography;

// export interface TubPlayerProps {
//   selectedDataPoint: TubDataPoint;
//   numDataPoints: number;
//   selectedDataPointIdx: number;
//   onChangeDataPoint: (idx: number) => void;
//   // data: TubDataPoint[];
// }

// export const TubPlayer: React.FunctionComponent<TubPlayerProps> = ({
//   selectedDataPoint,
//   numDataPoints,
//   onChangeDataPoint,
//   selectedDataPointIdx
// }) => {
//   const [isPlaying, setIsPlaying] = useState(false);

//   const onSelectedDataPointChanged = (newValue: SliderValue) =>
//     typeof newValue === "number" ? onChangeDataPoint(newValue - 1) : null;

//   const onTogglePlaying = () => setIsPlaying(!isPlaying);

//   useInterval(() => {
//     if (isPlaying) {
//       onChangeDataPoint((selectedDataPointIdx + 1) % numDataPoints);
//     }
//   }, 100);

//   return (
//     <div className={styles.tubPlayer}>
//       <img
//         className={styles.tubPlayerImage}
//         src={
//           selectedDataPoint &&
//           "http://localhost:8080" + selectedDataPoint.imageName
//         }
//       />

//       <div className={styles.tubplayerslidercontrol}>
//         <Slider
//           className={styles.tubplayerslider}
//           min={1}
//           max={numDataPoints}
//           value={selectedDataPointIdx + 1}
//           onChange={onSelectedDataPointChanged}
//         />
//         <div className={styles.tubplayerplaybutton}>
//           {isPlaying ? (
//             <Icon type="pause" onClick={onTogglePlaying} />
//           ) : (
//               <Icon type="caret-right" onClick={onTogglePlaying} />
//             )}
//         </div>
//       </div>

//       {/* <Row>
//         <Col xs={12}>
//           <Statistic title="Angle" value={selectedDataPoint.angle} />
//         </Col>
//         <Col xs={12}>
//           <Statistic title="Throttle" value={selectedDataPoint.throttle} />
//         </Col>
//       </Row> */}

//       <div className={styles.tubPlayerDetails}>
//         <div className={styles.tubPlayerDataPair}>
//           <Text className={styles.tubPlayerDataPairLabel}>Angle:</Text>
//           <Text>{selectedDataPoint && selectedDataPoint.angle}</Text>
//         </div>

//         <div className={styles.tubPlayerDataPair}>
//           <Text className={styles.tubPlayerDataPairLabel}>Throttle:</Text>
//           <Text>{selectedDataPoint && selectedDataPoint.throttle}</Text>
//         </div>
//       </div>
//     </div>
//   );
// };

// /**
//  * From: https://overreacted.io/making-setinterval-declarative-with-react-hooks/#just-show-me-the-code
//  */
// function useInterval(callback: () => void, delay: number) {
//   const savedCallback = useRef<() => void>();

//   // Remember the latest callback.
//   useEffect(() => {
//     savedCallback.current = callback;
//   }, [callback]);

//   // Set up the interval.
//   useEffect(() => {
//     function tick() {
//       savedCallback.current!();
//     }

//     if (delay !== null) {
//       let id = setInterval(tick, delay);
//       return () => clearInterval(id);
//     }
//   }, [delay]);
// }
