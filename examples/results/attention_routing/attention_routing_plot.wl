(* Attention Routing: Planning Attractor Boundary Plot *)
(* Plots strength sweep for Gemma 2 2B (426K, 2.5M CLTs) and Llama 3.2 1B (524K CLT) *)

SetDirectory[NotebookDirectory[]];

(* --- Load JSON data --- *)
data426k = Import["gemma-2-2b-426k.json", "RawJSON"];
data2m = Import["gemma-2-2b-2.5m.json", "RawJSON"];
dataLlama = Import["llama-3.2-1b-524k.json", "RawJSON"];

(* --- Extract strength sweep data --- *)
sweep426k = data426k["strength_sweep"];
sweep2m = data2m["strength_sweep"];

strengths426k = #["strength"] & /@ sweep426k;
topDelta426k = Abs[#["top_head_delta"]] & /@ sweep426k;
totalRouting426k = #["total_routing_shift"] & /@ sweep426k;

strengths2m = #["strength"] & /@ sweep2m;
topDelta2m = Abs[#["top_head_delta"]] & /@ sweep2m;
totalRouting2m = #["total_routing_shift"] & /@ sweep2m;

sweepLlama = dataLlama["strength_sweep"];
strengthsLlama = #["strength"] & /@ sweepLlama;
topDeltaLlama = Abs[#["top_head_delta"]] & /@ sweepLlama;
totalRoutingLlama = #["total_routing_shift"] & /@ sweepLlama;

(* --- Linear fit on first 6 points (strength 0-10) for 426K --- *)
fitData426k = Transpose[{strengths426k[[1 ;; 6]], topDelta426k[[1 ;; 6]]}];
linearFit426k = Fit[fitData426k, {1, x}, x];

(* --- Plot 1: Top Head Delta vs Strength (soft boundary) --- *)
topHeadPlot = Show[
  ListLinePlot[
    {
      Transpose[{strengths426k, topDelta426k}],
      Transpose[{strengths2m, topDelta2m}]
    },
    PlotStyle -> {
      {Thick, RGBColor[0.2, 0.4, 0.8]},
      {Thick, RGBColor[0.8, 0.3, 0.2]}
    },
    PlotMarkers -> {{\[FilledCircle], 8}, {\[FilledSquare], 8}},
    PlotLegends -> Placed[
      {"426K CLT (L22:10243)", "2.5M CLT (L25:82839)"},
      {0.35, 0.85}
    ],
    AxesLabel -> {"Steering Strength", "|Top Head \[CapitalDelta]Attn|"},
    PlotLabel -> Style[
      "Planning Attractor: Attention Routing vs Strength\n(Gemma 2 2B, suppress+inject, pos 23\[Rule]30)",
      14, Bold
    ],
    PlotRange -> All,
    ImageSize -> 600,
    GridLines -> Automatic,
    GridLinesStyle -> Directive[LightGray, Dashed]
  ],
  (* Linear extrapolation from strength 0-10 *)
  Plot[linearFit426k, {x, 0, 20},
    PlotStyle -> {Dashed, LightGray, Thickness[0.002]}
  ],
  (* Annotate the deviation region *)
  Graphics[{
    LightYellow, Opacity[0.3],
    Rectangle[{14, 0}, {20, Max[topDelta426k] * 1.1}]
  }],
  Graphics[{
    GrayLevel[0.4],
    Text[Style["saturation\nonset", 10, Italic], {17, Max[topDelta426k] * 0.5}]
  }]
];

(* --- Plot 1b: Cross-model strength sweep (Gemma 426K vs Llama 524K) --- *)
crossModelTopHead = Show[
  ListLinePlot[
    {
      Transpose[{strengths426k, topDelta426k}],
      Transpose[{strengthsLlama, topDeltaLlama}]
    },
    PlotStyle -> {
      {Thick, RGBColor[0.2, 0.4, 0.8]},
      {Thick, RGBColor[0.2, 0.7, 0.3]}
    },
    PlotMarkers -> {{\[FilledCircle], 8}, {\[FilledDiamond], 8}},
    PlotLegends -> Placed[
      {"Gemma 2 2B / 426K (L21:H5)", "Llama 3.2 1B / 524K (L13:H14)"},
      {0.4, 0.85}
    ],
    AxesLabel -> {"Steering Strength", "|Top Head \[CapitalDelta]Attn|"},
    PlotLabel -> Style[
      "Planning Routing: Cross-Model Comparison\n(suppress+inject, planning site \[Rule] output)",
      14, Bold
    ],
    PlotRange -> All,
    ImageSize -> 600,
    GridLines -> Automatic,
    GridLinesStyle -> Directive[LightGray, Dashed]
  ]
];

(* --- Plot 1c: Cross-model total routing shift --- *)
crossModelTotal = ListLinePlot[
  {
    Transpose[{strengths426k, totalRouting426k}],
    Transpose[{strengthsLlama, totalRoutingLlama}]
  },
  PlotStyle -> {
    {Thick, RGBColor[0.2, 0.4, 0.8]},
    {Thick, RGBColor[0.2, 0.7, 0.3]}
  },
  PlotMarkers -> {{\[FilledCircle], 8}, {\[FilledDiamond], 8}},
  PlotLegends -> Placed[
    {"Gemma 2 2B / 426K", "Llama 3.2 1B / 524K"},
    {0.3, 0.85}
  ],
  AxesLabel -> {"Steering Strength", "Total Routing Shift"},
  PlotLabel -> Style[
    "Total Attention Redistribution: Cross-Model\n(sum of |delta| across all heads)",
    14, Bold
  ],
  PlotRange -> All,
  ImageSize -> 600,
  GridLines -> Automatic,
  GridLinesStyle -> Directive[LightGray, Dashed]
];

(* --- Plot 2: Total Routing Shift vs Strength --- *)
totalRoutingPlot = ListLinePlot[
  {
    Transpose[{strengths426k, totalRouting426k}],
    Transpose[{strengths2m, totalRouting2m}]
  },
  PlotStyle -> {
    {Thick, RGBColor[0.2, 0.4, 0.8]},
    {Thick, RGBColor[0.8, 0.3, 0.2]}
  },
  PlotMarkers -> {{\[FilledCircle], 8}, {\[FilledSquare], 8}},
  PlotLegends -> Placed[
    {"426K CLT", "2.5M CLT"},
    {0.3, 0.85}
  ],
  AxesLabel -> {"Steering Strength", "Total Routing Shift"},
  PlotLabel -> Style[
    "Total Attention Redistribution vs Strength\n(sum of |delta| across all 208 heads)",
    14, Bold
  ],
  PlotRange -> All,
  ImageSize -> 600,
  GridLines -> Automatic,
  GridLinesStyle -> Directive[LightGray, Dashed]
];

(* --- Plot 3: Top 10 head deltas (426K, strength 10) --- *)
headDeltas426k = Flatten[data426k["head_deltas"], 1];
headDeltasSorted = SortBy[headDeltas426k, -Abs[#["delta"]] &];
top10 = headDeltasSorted[[1 ;; 10]];

headLabels = ("L" <> ToString[#["layer"]] <> ":H" <> ToString[#["head"]]) & /@ top10;
headValues = #["delta"] & /@ top10;

headBarPlot = BarChart[
  headValues,
  ChartLabels -> Placed[headLabels, Below],
  ChartStyle -> (If[# > 0, RGBColor[0.3, 0.7, 0.3], RGBColor[0.8, 0.3, 0.3]] & /@ headValues),
  AxesLabel -> {None, "\[CapitalDelta]Attention (pos 30\[Rule]23)"},
  PlotLabel -> Style[
    "Top 10 Routing Heads\n(426K CLT, suppress+inject, strength 10)",
    14, Bold
  ],
  ImageSize -> 600,
  GridLines -> {None, Automatic},
  GridLinesStyle -> Directive[LightGray, Dashed]
];

(* --- Plot 4: Top 10 head deltas (2.5M CLT, strength 10) --- *)
headDeltas2mAll = Flatten[data2m["head_deltas"], 1];
headDeltasSorted2m = SortBy[headDeltas2mAll, -Abs[#["delta"]] &];
topHeads2m = headDeltasSorted2m[[1 ;; 10]];

headLabels2m = ("L" <> ToString[#["layer"]] <> ":H" <> ToString[#["head"]]) & /@ topHeads2m;
headValues2m = N[#["delta"]] & /@ topHeads2m;

headBarPlot2m = BarChart[
  headValues2m,
  ChartLabels -> Placed[headLabels2m, Below],
  ChartStyle -> (If[# > 0, RGBColor[0.3, 0.7, 0.3], RGBColor[0.8, 0.3, 0.3]] & /@ headValues2m),
  AxesLabel -> {None, "\[CapitalDelta]Attention (pos 30\[Rule]23)"},
  PlotLabel -> Style[
    "Top 10 Routing Heads\n(2.5M CLT, suppress+inject, strength 10)",
    14, Bold
  ],
  ImageSize -> 600,
  GridLines -> {None, Automatic},
  GridLinesStyle -> Directive[LightGray, Dashed],
  PlotRange -> {Min[headValues] * 1.1, Max[headValues] * 1.1} (* 426K scale for comparison *)
];

(* --- Plot 5: Top 10 head deltas (Llama 3.2 1B / 524K CLT) --- *)
headDeltasLlama = Flatten[dataLlama["head_deltas"], 1];
headDeltasSortedLlama = SortBy[headDeltasLlama, -Abs[#["delta"]] &];
topHeadsLlama = headDeltasSortedLlama[[1 ;; 10]];

headLabelsLlama = ("L" <> ToString[#["layer"]] <> ":H" <> ToString[#["head"]]) & /@ topHeadsLlama;
headValuesLlama = N[#["delta"]] & /@ topHeadsLlama;

headBarPlotLlama = BarChart[
  headValuesLlama,
  ChartLabels -> Placed[headLabelsLlama, Below],
  ChartStyle -> (If[# > 0, RGBColor[0.3, 0.7, 0.3], RGBColor[0.8, 0.3, 0.3]] & /@ headValuesLlama),
  AxesLabel -> {None, "\[CapitalDelta]Attention (pos 29\[Rule]23)"},
  PlotLabel -> Style[
    "Top 10 Planning Routing Heads\n(Llama 3.2 1B, 524K CLT, suppress+inject, strength 15)",
    14, Bold
  ],
  ImageSize -> 600,
  GridLines -> {None, Automatic},
  GridLinesStyle -> Directive[LightGray, Dashed]
];

(* --- Export --- *)
exportDir = FileNameJoin[{Directory[], "plots"}];
If[!DirectoryQ[exportDir], CreateDirectory[exportDir]];

Export[FileNameJoin[{exportDir, "strength_sweep_top_head.png"}], topHeadPlot, ImageResolution -> 150];
Export[FileNameJoin[{exportDir, "strength_sweep_total_routing.png"}], totalRoutingPlot, ImageResolution -> 150];
Export[FileNameJoin[{exportDir, "top10_routing_heads.png"}], headBarPlot, ImageResolution -> 150];
Export[FileNameJoin[{exportDir, "top10_routing_heads_2.5m.png"}], headBarPlot2m, ImageResolution -> 150];
Export[FileNameJoin[{exportDir, "top10_routing_heads_llama.png"}], headBarPlotLlama, ImageResolution -> 150];
Export[FileNameJoin[{exportDir, "cross_model_top_head.png"}], crossModelTopHead, ImageResolution -> 150];
Export[FileNameJoin[{exportDir, "cross_model_total_routing.png"}], crossModelTotal, ImageResolution -> 150];

Print["Exported 7 plots to ", exportDir];
