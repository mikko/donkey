import { useEffect } from "react";

/**
 * From: https://github.com/imbhargav5/rooks/blob/master/packages/did-mount/src/index.js
 *
 * @param {function} callback Callback function to be called on mount
 */
export function useDidMount(callback: Function) {
  useEffect(() => {
    if (typeof callback === "function") {
      callback();
    }
  }, []);
}
