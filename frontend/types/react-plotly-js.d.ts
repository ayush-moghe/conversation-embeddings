declare module "react-plotly.js" {
  import { ComponentType, CSSProperties } from "react";
  import { Config, Data, Layout } from "plotly.js";

  export type PlotParams = {
    data: Data[];
    layout?: Partial<Layout>;
    config?: Partial<Config>;
    style?: CSSProperties;
    className?: string;
    useResizeHandler?: boolean;
  };

  const Plot: ComponentType<PlotParams>;
  export default Plot;
}