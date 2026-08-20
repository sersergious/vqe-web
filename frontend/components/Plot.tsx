"use client";

import { useEffect, useRef } from "react";

// Plotly is imported inside the effect so it never runs during SSR and stays
// out of the initial bundle.
export default function Plot({
  data,
  layout,
}: {
  data: unknown[];
  layout: Record<string, unknown>;
}) {
  const ref = useRef<HTMLDivElement>(null);

  // Callers pass fresh object literals every render, so this effect re-runs
  // often. Plotly.react diffs against the existing plot, so redrawing is cheap
  // — but purging here instead of on unmount would blank the chart each time.
  useEffect(() => {
    let cancelled = false;
    const element = ref.current;

    import("plotly.js-dist-min").then((module) => {
      if (cancelled || !element) return;
      module.default.react(
        element,
        data,
        {
          paper_bgcolor: "transparent",
          plot_bgcolor: "transparent",
          font: { color: "#c9d1d9", size: 12 },
          margin: { l: 60, r: 20, t: 44, b: 52 },
          legend: { orientation: "h", y: -0.22 },
          ...layout,
        },
        { displaylogo: false, responsive: true },
      );
    });

    return () => {
      cancelled = true;
    };
  }, [data, layout]);

  useEffect(() => {
    const element = ref.current;
    return () => {
      if (element) {
        import("plotly.js-dist-min").then((module) => module.default.purge(element));
      }
    };
  }, []);

  return <div ref={ref} className="plot" />;
}
