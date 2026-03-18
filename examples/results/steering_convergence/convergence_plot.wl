(* ::Package:: *)

(* ================================================================== *)
(* Steering Convergence: Attractor Dynamics in LLM Residual Streams   *)
(* candle-mi example: steering_convergence                            *)
(*                                                                     *)
(* Iterates over all *.json files in the notebook directory and       *)
(* produces 3 plots per model:                                        *)
(* 1. Convergence matrix heatmap (injection x observation layer)      *)
(* 2. P(target) bar chart by injection layer                          *)
(* 3. Strength sweep: P(target) + KL divergence (dual axis)          *)
(* ================================================================== *)

plotDir = FileNameJoin[{NotebookDirectory[], "plots"}];
If[!DirectoryQ[plotDir], CreateDirectory[plotDir]];

(* --- Discover all JSON files in this folder --- *)
jsonFiles = FileNames["*.json", NotebookDirectory[]];
Print["Found ", Length[jsonFiles], " JSON file(s): ", FileNameTake /@ jsonFiles];

(* ================================================================== *)
(* Process each JSON file                                              *)
(* ================================================================== *)

Do[
  Print["\n========================================"];
  Print["Processing: ", FileNameTake[jsonFile]];
  Print["========================================"];

  raw = Import[jsonFile, "RawJSON"];

  modelId   = raw["model_id"];
  nLayers   = raw["n_layers"];
  threshold = raw["threshold"];
  matrix    = raw["convergence_matrix"];
  summaries = raw["layer_summaries"];
  sweep     = raw["strength_sweep"];
  bestLayer = raw["best_injection_layer"];
  prompt    = raw["prompt"];
  target    = raw["target_token"];
  baselineP = raw["baseline_p_target"];

  Print["Model: ", modelId];
  Print["Layers: ", nLayers, ", threshold: ", threshold];
  Print["Best injection layer: ", bestLayer];
  Print["Baseline P(\"", target, "\"): ",
    NumberForm[baselineP * 100, 3], "%"];

  prefix = StringReplace[modelId, "/" -> "_"] <> "_";

  (* ================================================================ *)
  (* PLOT 1: Convergence Matrix Heatmap                               *)
  (*                                                                   *)
  (* Diverging colormap: dark (low similarity) -> white (1.0).        *)
  (* Mesh grid for readability. Range 0.90-1.0 for contrast.         *)
  (* ================================================================ *)

  heatmap = MatrixPlot[
    Reverse[matrix],  (* Reverse so layer 0 is at top *)
    ColorFunction -> (Blend[
      {Darker[Blue], Cyan, Green, Yellow, Orange, Darker[Red]},
      Rescale[#, {0.90, 1.0}]] &),
    ColorFunctionScaling -> False,
    PlotRange -> {0.90, 1.001},
    ClippingStyle -> {Darker[Blue], Darker[Red]},
    Mesh -> True,
    MeshStyle -> Directive[GrayLevel[0.3], Thin],
    PlotLegends -> BarLegend[
      {Blend[
        {Darker[Blue], Cyan, Green, Yellow, Orange, Darker[Red]},
        Rescale[#, {0.90, 1.0}]] &,
       {0.90, 1.0}},
      LegendLabel -> "Cosine\nSimilarity",
      LabelStyle -> {10},
      LegendMarkerSize -> 250
    ],
    FrameLabel -> {
      Style["Injection Layer", 12],
      Style["Observation Layer", 12]
    },
    FrameTicks -> {
      {Table[{nLayers - i, i}, {i, 0, nLayers - 1}], None},
      {Table[{i + 1, i}, {i, 0, nLayers - 1}], None}
    },
    PlotLabel -> Style[
      "Steering Convergence Matrix\n" <> modelId,
      14, Bold
    ],
    ImageSize -> 650,
    AspectRatio -> 1
  ];

  Export[
    FileNameJoin[{plotDir, prefix <> "convergence_matrix.png"}],
    heatmap, ImageResolution -> 200
  ];

  (* ================================================================ *)
  (* PLOT 2: P(target) by Injection Layer                             *)
  (* ================================================================ *)

  pTargets = #["p_target"] & /@ summaries;

  pTargetPlot = BarChart[
    pTargets * 100,
    ChartLabels -> Range[0, nLayers - 1],
    AxesLabel -> {Style["Layer", 11], Style["P(target) %", 11]},
    PlotLabel -> Style[
      "P(\"" <> target <> "\") by Injection Layer\n" <> modelId,
      12, Bold
    ],
    PlotRange -> {0, Max[pTargets] * 120},
    GridLines -> {None, {baselineP * 100}},
    ImageSize -> 650,
    ChartStyle -> Directive[Opacity[0.7], Blue],
    Epilog -> {
      Dashed, Red, Thick,
      InfiniteLine[{0, baselineP * 100}, {1, 0}],
      Inset[
        Style["baseline", 10, Red],
        {1.5, baselineP * 100 + 1.5}
      ]
    }
  ];

  Export[
    FileNameJoin[{plotDir, prefix <> "p_target_by_layer.png"}],
    pTargetPlot, ImageResolution -> 200
  ];

  (* ================================================================ *)
  (* PLOT 3: Strength Sweep — dual axis (P(target) + KL divergence)  *)
  (* ================================================================ *)

  strengths = #["strength"] & /@ sweep;
  sweepP = #["p_target"] & /@ sweep;
  sweepKL = #["kl_divergence"] & /@ sweep;

  (* Left axis: P(target) *)
  pCurve = ListLinePlot[
    Transpose[{strengths, sweepP * 100}],
    PlotRange -> {Automatic, {0, Automatic}},
    PlotStyle -> {Thick, Blue},
    PlotMarkers -> {Automatic, 8},
    Frame -> True,
    FrameLabel -> {
      {Style["P(target) %", 12, Blue], Style["KL Divergence", 12, Orange]},
      {Style["Strength", 12], None}
    },
    FrameTicks -> {{Automatic, None}, {Automatic, None}},
    FrameStyle -> {{Blue, Orange}, {Black, Black}},
    PlotLabel -> Style[
      "Strength Sweep at Layer " <> ToString[bestLayer] <>
      "\n" <> modelId,
      13, Bold
    ],
    ImageSize -> 600,
    GridLines -> {None, {baselineP * 100}},
    Epilog -> {
      Dashed, Red, Thin,
      InfiniteLine[{0, baselineP * 100}, {1, 0}]
    }
  ];

  (* Right axis: KL divergence — rescale to share vertical space *)
  maxP = Max[sweepP * 100];
  maxKL = Max[sweepKL];
  klScale = If[maxKL > 0, maxP / maxKL, 1];

  klCurve = ListLinePlot[
    Transpose[{strengths, sweepKL * klScale}],
    PlotStyle -> {Thick, Orange},
    PlotMarkers -> {Automatic, 8}
  ];

  (* Overlay P(target) curve with rescaled KL curve *)
  strengthCombo = Show[
    pCurve,
    klCurve,
    Frame -> True,
    FrameLabel -> {
      {Style["P(target) %", 12, Blue],
       Style["KL Divergence", 12, Orange]},
      {Style["Strength", 12], None}
    },
    FrameTicks -> {
      {Automatic,
       Table[{v * klScale, NumberForm[v, {3, 1}]},
         {v, 0, maxKL, maxKL / 5}]},
      {Automatic, None}
    },
    FrameStyle -> {{Blue, Orange}, {Black, Black}},
    ImageSize -> 600,
    PlotLabel -> Style[
      "Strength Sweep at Layer " <> ToString[bestLayer] <>
      "\n" <> modelId <>
      "\nBlue = P(target), Orange = KL Divergence",
      12, Bold
    ]
  ];

  Export[
    FileNameJoin[{plotDir, prefix <> "strength_sweep.png"}],
    strengthCombo, ImageResolution -> 200
  ];

  Print["Exported 3 plots for ", modelId];

  (* Display *)
  Print@Column[{
    Style["Steering Convergence Analysis", 18, Bold],
    Style[modelId, 14],
    Style["Prompt: \"" <> prompt <> "\"  Target: \"" <> target <> "\"", 11],
    Spacer[10],
    heatmap,
    Spacer[10],
    pTargetPlot,
    Spacer[10],
    strengthCombo
  }];

, {jsonFile, jsonFiles}];

Print["\n=== Done. Processed ", Length[jsonFiles], " model(s). ==="];
