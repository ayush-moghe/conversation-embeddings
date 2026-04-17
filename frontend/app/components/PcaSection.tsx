"use client";

import dynamic from "next/dynamic";
import { ChangeEvent, ComponentType, useEffect, useMemo, useRef, useState } from "react";
import { Config, Data, Layout } from "plotly.js";
import { PlotParams } from "react-plotly.js";

type PcaItem = {
  filename: string;
  embedding: number[];
  point: number[];
  clusterLabel: number;
};

type BatchEmbeddingPcaResponse = {
  items: PcaItem[];
  dimensions: number;
  count: number;
  clusterCount: number;
  pcaDimensions: number;
};

type PcaSectionProps = {
  apiBaseUrl: string;
};

const Plot = dynamic(() => import("react-plotly.js"), {
  ssr: false,
}) as ComponentType<PlotParams>;

const REFRESH_ICON = {
  width: 1000,
  height: 1000,
  path: "M500 120c209.9 0 380 170.1 380 380S709.9 880 500 880 120 709.9 120 500h110c0 149.1 120.9 270 270 270s270-120.9 270-270-120.9-270-270-270c-87.3 0-165 41.6-214.3 106.1L360 410H130V180l80.8 80.8C279.2 173.1 382.9 120 500 120z",
};

const PCA_STORAGE_KEY = "conversation-embeddings:pca-state";

export default function PcaSection({ apiBaseUrl }: PcaSectionProps) {
  const [pcaMode, setPcaMode] = useState<2 | 3>(3);
  const [viewResetToken, setViewResetToken] = useState(0);
  const [folderLoading, setFolderLoading] = useState(false);
  const [folderError, setFolderError] = useState<string | null>(null);
  const [folderStats, setFolderStats] = useState<string | null>(null);
  const [pcaItems, setPcaItems] = useState<PcaItem[]>([]);
  const [resultDimension, setResultDimension] = useState<2 | 3 | null>(null);
  const [isHydrated, setIsHydrated] = useState(false);
  const folderInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(PCA_STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw) as {
          pcaMode?: 2 | 3;
          folderStats?: string | null;
          pcaItems?: PcaItem[];
          resultDimension?: 2 | 3 | null;
        };

        if (parsed.pcaMode === 2 || parsed.pcaMode === 3) {
          setPcaMode(parsed.pcaMode);
        }
        if (parsed.resultDimension === 2 || parsed.resultDimension === 3) {
          setResultDimension(parsed.resultDimension);
        }
        setFolderStats(parsed.folderStats ?? null);
        setPcaItems(Array.isArray(parsed.pcaItems) ? parsed.pcaItems : []);
      }
    } catch {
      // Ignore corrupt localStorage values and continue with defaults.
    } finally {
      setIsHydrated(true);
    }
  }, []);

  useEffect(() => {
    if (!isHydrated) {
      return;
    }

    const payload = {
      pcaMode,
      folderStats,
      pcaItems,
      resultDimension,
    };

    localStorage.setItem(PCA_STORAGE_KEY, JSON.stringify(payload));
  }, [isHydrated, pcaMode, folderStats, pcaItems, resultDimension]);

  useEffect(() => {
    if (!folderInputRef.current) {
      return;
    }

    folderInputRef.current.setAttribute("webkitdirectory", "");
    folderInputRef.current.setAttribute("directory", "");
  }, []);

  const handleFolderUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    if (files.length === 0) {
      return;
    }

    setFolderLoading(true);
    setFolderError(null);
    setFolderStats(null);
    setPcaItems([]);

    try {
      const txtFiles = files.filter((file) => file.name.toLowerCase().endsWith(".txt"));

      if (txtFiles.length === 0) {
        setFolderError("No .txt files were found in the selected folder.");
        return;
      }

      const documents = await Promise.all(
        txtFiles.map(async (file) => {
          const content = await file.text();
          return {
            filename: file.webkitRelativePath || file.name,
            conversation: content,
          };
        }),
      );

      const response = await fetch(`${apiBaseUrl}/api/getEmbeddingsPca`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ documents, pcaDimensions: pcaMode }),
      });

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }

      const data: BatchEmbeddingPcaResponse = await response.json();
      setPcaItems(data.items ?? []);
      setResultDimension(data.pcaDimensions === 2 ? 2 : 3);
      setFolderStats(
        `${data.count ?? 0} files processed${data.dimensions ? ` • ${data.dimensions} embedding dimensions` : ""}${typeof data.clusterCount === "number" ? ` • ${data.clusterCount} clusters` : ""}${data.pcaDimensions ? ` • ${data.pcaDimensions}D PCA` : ""}`,
      );
    } catch (requestError) {
      const fallback = "Could not generate batch embeddings and PCA.";
      setFolderError(
        requestError instanceof Error ? `${fallback} ${requestError.message}` : fallback,
      );
    } finally {
      setFolderLoading(false);
      event.target.value = "";
    }
  };

  const chartData = useMemo<Data[]>(() => {
    if (pcaItems.length === 0) {
      return [];
    }

    const palette = [
      "#e63946",
      "#1d3557",
      "#ff7f11",
      "#2a9d8f",
      "#6a4c93",
      "#118ab2",
      "#ef476f",
      "#06d6a0",
      "#3a86ff",
      "#ffbe0b",
      "#8338ec",
      "#fb5607",
    ];

    const groupedItems = new Map<number, PcaItem[]>();
    pcaItems.forEach((item) => {
      const key = item.clusterLabel;
      const bucket = groupedItems.get(key) ?? [];
      bucket.push(item);
      groupedItems.set(key, bucket);
    });

    const sortedLabels = Array.from(groupedItems.keys()).sort((a, b) => a - b);

    return sortedLabels.map((label, index) => {
      const items = groupedItems.get(label) ?? [];
      const isNoise = label < 0;

      if ((resultDimension ?? pcaMode) === 2) {
        return {
          x: items.map((item) => item.point[0]),
          y: items.map((item) => item.point[1]),
          text: items.map((item) => item.filename),
          type: "scatter",
          mode: "markers",
          name: isNoise ? "Noise / Unclustered" : `Cluster ${label + 1}`,
          marker: {
            size: isNoise ? 8 : 10,
            color: isNoise ? "#7f8a84" : palette[index % palette.length],
            opacity: isNoise ? 0.58 : 0.88,
          },
        };
      }

      return {
        x: items.map((item) => item.point[0]),
        y: items.map((item) => item.point[1]),
        z: items.map((item) => item.point[2] ?? 0),
        text: items.map((item) => item.filename),
        type: "scatter3d",
        mode: "markers",
        name: isNoise ? "Noise / Unclustered" : `Cluster ${label + 1}`,
        marker: {
          size: isNoise ? 5 : 7,
          color: isNoise ? "#7f8a84" : palette[index % palette.length],
          opacity: isNoise ? 0.58 : 0.88,
        },
      };
    });
  }, [pcaItems, pcaMode, resultDimension]);

  const axisRanges = useMemo(() => {
    if (pcaItems.length === 0) {
      return {
        x: [-1, 1] as [number, number],
        y: [-1, 1] as [number, number],
        z: [-1, 1] as [number, number],
      };
    }

    const xs = pcaItems.map((item) => item.point[0]);
    const ys = pcaItems.map((item) => item.point[1]);
    const zs = pcaItems.map((item) => item.point[2] ?? 0);

    const withPadding = (values: number[]): [number, number] => {
      const min = Math.min(...values);
      const max = Math.max(...values);
      const span = Math.max(max - min, 1e-6);
      const pad = span * 0.12;
      return [min - pad, max + pad];
    };

    return {
      x: withPadding(xs),
      y: withPadding(ys),
      z: withPadding(zs),
    };
  }, [pcaItems]);

  const chartLayout = useMemo<Partial<Layout>>(
    () => {
      if ((resultDimension ?? pcaMode) === 2) {
        return {
          title: { text: "" },
          autosize: true,
          uirevision: "pca-2d-locked-axes",
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          margin: { l: 50, r: 20, b: 50, t: 10 },
          dragmode: "pan",
          xaxis: { title: { text: "PCA X" }, autorange: false, range: axisRanges.x },
          yaxis: { title: { text: "PCA Y" }, autorange: false, range: axisRanges.y },
          legend: { orientation: "h", y: 1.08 },
        };
      }

      return {
        title: { text: "" },
        autosize: true,
        uirevision: "pca-3d-locked-axes",
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        margin: { l: 0, r: 0, b: 0, t: 0 },
        scene: {
          dragmode: "turntable",
          aspectmode: "cube",
          camera: {
            up: { x: 0, y: 0, z: 1 },
          },
          xaxis: { title: { text: "PCA X" }, autorange: false, range: axisRanges.x },
          yaxis: { title: { text: "PCA Y" }, autorange: false, range: axisRanges.y },
          zaxis: { title: { text: "PCA Z" }, autorange: false, range: axisRanges.z },
        },
      };
    },
    [axisRanges, pcaMode, resultDimension],
  );

  const chartConfig = useMemo<Partial<Config>>(
    () => {
      const currentDimension = resultDimension ?? pcaMode;

      if (currentDimension === 2) {
        return {
          responsive: true,
          displaylogo: false,
          scrollZoom: true,
          modeBarButtons: [
            [
              {
                name: "Reset View",
                title: "Reset View",
                icon: REFRESH_ICON,
                click: (gd: unknown) => {
                  void gd;
                  setViewResetToken((value) => value + 1);
                },
              },
              "zoomIn2d",
              "zoomOut2d",
              "pan2d",
            ],
          ],
        } as Partial<Config>;
      }

      return {
        responsive: true,
        displaylogo: false,
        scrollZoom: true,
        modeBarButtons: [
          [
            {
              name: "Reset View",
              title: "Reset View",
              icon: REFRESH_ICON,
              click: (gd: unknown) => {
                void gd;
                setViewResetToken((value) => value + 1);
              },
            },
            "zoom3d",
            "pan3d",
          ],
        ],
      } as Partial<Config>;
    },
    [resultDimension, pcaMode],
  );

  const showHeaderStats = !!folderStats && (!resultDimension || resultDimension === pcaMode);
  const plotKey = `pca-${resultDimension ?? pcaMode}-${viewResetToken}`;

  return (
    <section className="result-card" aria-live="polite" id="folder-pca">
      <div className="result-header">
        <h2>Folder PCA ({pcaMode}D)</h2>
        {showHeaderStats ? <span>{folderStats}</span> : null}
      </div>

      <div className="mode-toggle" role="group" aria-label="PCA dimensionality">
        <span className="mode-label">Dimension</span>
        <button
          type="button"
          className={`mode-button ${pcaMode === 2 ? "active" : ""}`}
          onClick={() => setPcaMode(2)}
        >
          2
        </button>
        <button
          type="button"
          className={`mode-button ${pcaMode === 3 ? "active" : ""}`}
          onClick={() => setPcaMode(3)}
        >
          3
        </button>
      </div>

      <div className="file-row">
        <label htmlFor="pcaFolderUpload" className="upload-button">
          Upload Folder of .txt
        </label>
        <input
          id="pcaFolderUpload"
          ref={folderInputRef}
          type="file"
          multiple
          onChange={handleFolderUpload}
        />
        <span className="file-label">
          {folderLoading ? `Generating ${pcaMode}D vectors and PCA...` : "Select a folder"}
        </span>
      </div>

      {folderError ? <p className="status error">{folderError}</p> : null}

      {chartData.length > 0 && resultDimension === pcaMode ? (
        <div className="chart-wrap">
          <Plot
            key={plotKey}
            data={chartData}
            layout={chartLayout}
            config={chartConfig}
            style={{ width: "100%", height: "100%" }}
          />
        </div>
      ) : (
        <p className="empty-state">
          {resultDimension && resultDimension !== pcaMode
            ? `Current graph is ${resultDimension}D. Upload again to generate ${pcaMode}D.`
            : `Upload a folder containing .txt files to visualize each file embedding in PCA ${pcaMode}D.`}
        </p>
      )}
    </section>
  );
}
