(* ================================================================== *)
(* Figure 13 Replication: Suppress + Inject Position Sweep            *)
(* candle-mi example: figure13_planning_poems                         *)
(* ================================================================== *)

(* Shared helper: extract sweep data from a JSON association *)
extractSweep[d_] := Module[{sw = d["sweep"]},
  <|
    "positions" -> sw[[All, "position"]],
    "probs"     -> sw[[All, "prob"]],
    "tokens"    -> sw[[All, "token"]],
    "baseline"  -> d["baseline_prob"]
  |>
];

(* Shared helper: clean token labels for display *)
cleanToken[t_] := StringReplace[t, {
  "\n" -> "\\n",
  " " -> "\[ThinSpace]"
}];

(* Shared helper: build styled bar data (first=dark gray, last=red, rest=gray) *)
styledBars[allProbs_, nAll_] := MapIndexed[
  Style[#1, Which[
    First[#2] == 1, GrayLevel[0.4],
    First[#2] == nAll, Darker[Red],
    True, GrayLevel[0.6]
  ]] &,
  allProbs
];

(* ================================================================== *)
(* PART 1: Llama 3.2 1B                                               *)
(* ================================================================== *)

llamaData = Import[
  FileNameJoin[{NotebookDirectory[], "llama_output.json"}],
  "RawJSON"
];

llamaSweep = extractSweep[llamaData];
llamaProbs = llamaSweep["probs"];
llamaTokens = llamaSweep["tokens"];
llamaBaseline = llamaSweep["baseline"];
llamaLabels = cleanToken /@ llamaTokens;

(* Prepend baseline as "No Steering" bar *)
llamaAllProbs = Prepend[llamaProbs, llamaBaseline];
llamaAllLabels = Prepend[llamaLabels, "No Steering"];
nLlama = Length[llamaAllProbs];

(* Clean token labels for display: escape special chars *)
cleanToken[t_] := StringReplace[t, {
  "\n" -> "\\n",
  " " -> "\[ThinSpace]"  (* leading spaces rendered as thin space *)
}];
tokenLabels = cleanToken /@ tokens;

(* --- Llama: linear scale --- *)
llamaLinear = BarChart[styledBars[llamaAllProbs, nLlama],
  ChartLabels -> Placed[
    Style[#, 6, FontFamily -> "Consolas"] & /@ llamaAllLabels,
    Below, Rotate[#, Pi/4] &
  ],
  PlotLabel -> Style[
    "Figure 13: Suppress \"" <> llamaData["suppress_word"] <>
    "\" + Inject \"" <> llamaData["inject_word"] <>
    "\"\n" <> llamaData["model"] <> " | strength=" <>
    ToString[llamaData["strength"]],
    12, Bold
  ],
  FrameLabel -> {
    Style["Token position", 11],
    Style["P(\"" <> llamaData["inject_word"] <> "\")", 11]
  },
  Frame -> True,
  ImageSize -> 900,
  AspectRatio -> 1/3,
  ImagePadding -> {{60, 20}, {120, 40}},
  PlotRangePadding -> {{Scaled[0.01], Scaled[0.01]}, {0, Scaled[0.05]}},
  GridLines -> {{{1.5, Directive[Dashed, GrayLevel[0.5]]}}, None}
];

(* --- Llama: log scale --- *)
llamaLog = BarChart[styledBars[llamaAllProbs, nLlama],
  ChartLabels -> Placed[
    Style[#, 6, FontFamily -> "Consolas"] & /@ llamaAllLabels,
    Below, Rotate[#, Pi/4] &
  ],
  ScalingFunctions -> "Log",
  PlotLabel -> Style[
    "Figure 13 (log): Suppress \"" <> llamaData["suppress_word"] <>
    "\" + Inject \"" <> llamaData["inject_word"] <>
    "\"\n" <> llamaData["model"],
    12, Bold
  ],
  FrameLabel -> {
    Style["Token position", 11],
    Style["P(\"" <> llamaData["inject_word"] <> "\") [log]", 11]
  },
  Frame -> True,
  ImageSize -> 900,
  AspectRatio -> 1/3,
  ImagePadding -> {{60, 20}, {120, 40}},
  PlotRangePadding -> {{Scaled[0.01], Scaled[0.01]}, {0, Scaled[0.05]}},
  GridLines -> {{{1.5, Directive[Dashed, GrayLevel[0.5]]}}, None}
];

(* Export Llama PNGs *)
Export[
  FileNameJoin[{NotebookDirectory[], "llama_linear.png"}],
  llamaLinear, ImageResolution -> 200
];
Export[
  FileNameJoin[{NotebookDirectory[], "llama_log.png"}],
  llamaLog, ImageResolution -> 200
];
Print["Exported llama_linear.png and llama_log.png"];

(* ================================================================== *)
(* PART 2: Gemma 2 2B                                                 *)
(* ================================================================== *)

gemmaData = Import[
  FileNameJoin[{NotebookDirectory[], "gemma_output.json"}],
  "RawJSON"
];

gemmaSweep = extractSweep[gemmaData];
gemmaProbs = gemmaSweep["probs"];
gemmaTokens = gemmaSweep["tokens"];
gemmaBaseline = gemmaSweep["baseline"];
gemmaLabels = cleanToken /@ gemmaTokens;

(* Prepend baseline as "No Steering" bar *)
gemmaAllProbs = Prepend[gemmaProbs, gemmaBaseline];
gemmaAllLabels = Prepend[gemmaLabels, "No Steering"];
nGemma = Length[gemmaAllProbs];

(* --- Gemma: linear scale --- *)
gemmaLinear = BarChart[styledBars[gemmaAllProbs, nGemma],
  ChartLabels -> Placed[
    Style[#, 6, FontFamily -> "Consolas"] & /@ gemmaAllLabels,
    Below, Rotate[#, Pi/4] &
  ],
  PlotLabel -> Style[
    "Figure 13: Suppress \"" <> gemmaData["suppress_word"] <>
    "\" + Inject \"" <> gemmaData["inject_word"] <>
    "\"\n" <> gemmaData["model"] <> " | strength=" <>
    ToString[gemmaData["strength"]],
    12, Bold
  ],
  FrameLabel -> {
    Style["Token position", 11],
    Style["P(\"" <> gemmaData["inject_word"] <> "\")", 11]
  },
  Frame -> True,
  ImageSize -> 900,
  AspectRatio -> 1/3,
  ImagePadding -> {{60, 20}, {120, 40}},
  PlotRangePadding -> {{Scaled[0.01], Scaled[0.01]}, {0, Scaled[0.05]}},
  GridLines -> {{{1.5, Directive[Dashed, GrayLevel[0.5]]}}, None}
];

(* --- Gemma: log scale --- *)
gemmaLog = BarChart[styledBars[gemmaAllProbs, nGemma],
  ChartLabels -> Placed[
    Style[#, 6, FontFamily -> "Consolas"] & /@ gemmaAllLabels,
    Below, Rotate[#, Pi/4] &
  ],
  ScalingFunctions -> "Log",
  PlotLabel -> Style[
    "Figure 13 (log): Suppress \"" <> gemmaData["suppress_word"] <>
    "\" + Inject \"" <> gemmaData["inject_word"] <>
    "\"\n" <> gemmaData["model"],
    12, Bold
  ],
  FrameLabel -> {
    Style["Token position", 11],
    Style["P(\"" <> gemmaData["inject_word"] <> "\") [log]", 11]
  },
  Frame -> True,
  ImageSize -> 900,
  AspectRatio -> 1/3,
  ImagePadding -> {{60, 20}, {120, 40}},
  PlotRangePadding -> {{Scaled[0.01], Scaled[0.01]}, {0, Scaled[0.05]}},
  GridLines -> {{{1.5, Directive[Dashed, GrayLevel[0.5]]}}, None}
];

(* Export Gemma PNGs *)
Export[
  FileNameJoin[{NotebookDirectory[], "gemma_linear.png"}],
  gemmaLinear, ImageResolution -> 200
];
Export[
  FileNameJoin[{NotebookDirectory[], "gemma_log.png"}],
  gemmaLog, ImageResolution -> 200
];
Print["Exported gemma_linear.png and gemma_log.png"];

(* ================================================================== *)
(* Display all four figures                                            *)
(* ================================================================== *)

Column[{
  Style["Llama 3.2 1B", 14, Bold], llamaLinear, llamaLog,
  Spacer[20],
  Style["Gemma 2 2B", 14, Bold], gemmaLinear, gemmaLog
}]

(* ================================================================== *)
(* Summary table                                                       *)
(* ================================================================== *)

Print["\n=== Comparison ==="];
Print["                    Llama 3.2 1B          Gemma 2 2B"];
Print["Baseline:           ",
  ScientificForm[llamaBaseline, 4], "          ",
  ScientificForm[gemmaBaseline, 4]];
Print["Max P:              ",
  NumberForm[Max[llamaProbs], 6], "          ",
  NumberForm[Max[gemmaProbs], 6]];
Print["Ratio:              ",
  NumberForm[Max[llamaProbs] / llamaBaseline, {6, 1}], "x          ",
  NumberForm[Max[gemmaProbs] / gemmaBaseline, {6, 1}], "x"];
