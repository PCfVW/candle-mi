(* ================================================================== *)
(* Character Count Helix: 3D scatter, cosine heatmap, variance bars   *)
(* candle-mi example: character_count_helix                           *)
(*                                                                    *)
(* Accepts both single-layer JSON (from --pca-layers) and sweep-mode  *)
(* JSON arrays (from --sweep). For sweep files, generates one set of  *)
(* plots per layer.                                                   *)
(* ================================================================== *)

(* --- Import JSON output from the Rust example --- *)
raw = Import[
  FileNameJoin[{NotebookDirectory[], "sweep.json"}],
  "RawJSON"
];

(* Normalise: wrap a single object in a list so the rest of the code
   always iterates over a list of layer entries. *)
layers = If[ListQ[raw] && MatchQ[raw, {__Association}],
  raw,
  {raw}
];

Print["Loaded ", Length[layers], " layer(s) from JSON."];

(* ================================================================== *)
(* Helper: generate all plots for one layer entry                     *)
(* ================================================================== *)

plotLayer[entry_Association] := Module[
  {modelId, layer, maxCC, evr, totalVar, projs, cosineSim,
   charCounts, pcCoords, helixPoints, helix3D, helixWithLegend,
   cosineMatrix, heatmap, varianceBars, helix456, helix456WithLegend,
   prefix},

  modelId   = entry["model_id"];
  layer     = entry["layer"];
  maxCC     = entry["max_char_count"];
  evr       = entry["explained_variance"];
  totalVar  = entry["total_variance_top6"];
  projs     = entry["projections"];
  cosineSim = entry["cosine_similarity"];

  charCounts = projs[[All, "char_count"]];
  pcCoords   = projs[[All, "pc"]];

  (* -------------------------------------------------------------- *)
  (* PLOT 1: 3D Helix — PC1 vs PC2 vs PC3, colored by char count    *)
  (* -------------------------------------------------------------- *)

  helix3D = ListPointPlot3D[
    {#[[1]], #[[2]], #[[3]]} & /@ pcCoords,
    ColorFunction -> (ColorData["Rainbow"][Rescale[#3, {0, 1}]] &),
    PlotStyle -> PointSize[Medium],
    AxesLabel -> {"PC1", "PC2", "PC3"},
    PlotLabel -> Style[
      "Character Count Helix\n" <> modelId <> " layer " <>
      ToString[layer],
      14, Bold
    ],
    ImageSize -> 600,
    Boxed -> True,
    BoxRatios -> {1, 1, 1}
  ];

  helixWithLegend = Legended[helix3D,
    BarLegend[{"Rainbow", {Min[charCounts], Max[charCounts]}},
      LegendLabel -> "Char count"
    ]
  ];

  (* -------------------------------------------------------------- *)
  (* PLOT 2: Cosine Similarity Heatmap                               *)
  (* -------------------------------------------------------------- *)

  cosineMatrix = ArrayReshape[
    cosineSim, {Length[cosineSim], Length[cosineSim]}
  ];

  heatmap = MatrixPlot[cosineMatrix,
    ColorFunction -> "TemperatureMap",
    PlotLegends -> Automatic,
    FrameLabel -> {"Char count index", "Char count index"},
    PlotLabel -> Style[
      "Cosine Similarity (ringing pattern)\n" <> modelId <>
      " layer " <> ToString[layer],
      12, Bold
    ],
    ImageSize -> 500
  ];

  (* -------------------------------------------------------------- *)
  (* PLOT 3: Explained Variance Bar Chart                            *)
  (* -------------------------------------------------------------- *)

  varianceBars = BarChart[100 * evr,
    ChartLabels -> Table["PC" <> ToString[i], {i, Length[evr]}],
    PlotLabel -> Style[
      "Explained Variance per Component\nTotal top-" <>
      ToString[Length[evr]] <> ": " <>
      ToString[NumberForm[100 totalVar, {4, 1}]] <> "%",
      12, Bold
    ],
    FrameLabel -> {None, "Variance (%)"},
    Frame -> True,
    ImageSize -> 400,
    ChartStyle -> Lighter[Blue]
  ];

  (* -------------------------------------------------------------- *)
  (* PLOT 4: PC4 vs PC5 vs PC6 (secondary twist)                    *)
  (* -------------------------------------------------------------- *)

  helix456WithLegend = If[Length[First[pcCoords]] >= 6,
    helix456 = ListPointPlot3D[
      {#[[4]], #[[5]], #[[6]]} & /@ pcCoords,
      ColorFunction -> (ColorData["Rainbow"][Rescale[#3, {0, 1}]] &),
      PlotStyle -> PointSize[Medium],
      AxesLabel -> {"PC4", "PC5", "PC6"},
      PlotLabel -> Style[
        "Secondary Twist (PC4-6)\n" <> modelId <>
        " layer " <> ToString[layer],
        14, Bold
      ],
      ImageSize -> 600,
      Boxed -> True,
      BoxRatios -> {1, 1, 1}
    ];
    Legended[helix456,
      BarLegend[{"Rainbow", {Min[charCounts], Max[charCounts]}},
        LegendLabel -> "Char count"
      ]
    ],
    Nothing
  ];

  (* -------------------------------------------------------------- *)
  (* Export PNGs to plots/ subfolder                                  *)
  (* -------------------------------------------------------------- *)

  prefix = "L" <> ToString[layer] <> "_";
  plotDir = FileNameJoin[{NotebookDirectory[], "plots"}];
  If[!DirectoryQ[plotDir], CreateDirectory[plotDir]];

  Export[
    FileNameJoin[{plotDir, prefix <> "helix_pc123.png"}],
    helixWithLegend, ImageResolution -> 200
  ];
  Export[
    FileNameJoin[{plotDir, prefix <> "cosine_heatmap.png"}],
    heatmap, ImageResolution -> 200
  ];
  Export[
    FileNameJoin[{plotDir, prefix <> "variance_bars.png"}],
    varianceBars, ImageResolution -> 200
  ];
  If[helix456WithLegend =!= Nothing,
    Export[
      FileNameJoin[{plotDir, prefix <> "helix_pc456.png"}],
      helix456WithLegend, ImageResolution -> 200
    ];
  ];

  Print["Layer ", layer, ": exported ", prefix, "*.png"];

  (* Return a column of plots for display *)
  Column[{
    Style["Layer " <> ToString[layer], 16, Bold],
    Style[modelId <> " | " <>
      ToString[NumberForm[100 totalVar, {4, 1}]] <> "% variance (top-6)",
      12],
    Spacer[10],
    helixWithLegend,
    Spacer[10],
    If[helix456WithLegend =!= Nothing, helix456WithLegend, Nothing],
    Spacer[10],
    Row[{varianceBars, Spacer[20], heatmap}]
  }]
];

(* ================================================================== *)
(* Generate plots for every layer in the file                         *)
(* ================================================================== *)

allPlots = plotLayer /@ layers;

(* Display all layers *)
Column[
  Prepend[
    Riffle[allPlots, Spacer[30]],
    Style["Character Count Helix Analysis", 20, Bold]
  ]
]
