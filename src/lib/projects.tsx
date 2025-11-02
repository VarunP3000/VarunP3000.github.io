import type { Project as CardProject } from "../components/ProjectCard";

// Extended shape used by the case study page (CardProject is still fine for cards)
type CaseStudyProject = CardProject & {
  problem?: string;
  approach?: string | string[];
  results?: string | string[];
  limitations?: string;
  nextSteps?: string | string[];
  year: number | string;
};

const slug = (x: string) =>
  x
    .toLowerCase()
    .replace(/&/g, "and")
    .replace(/→/g, "-")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "");

export const projects: CaseStudyProject[] = [
  // --- Personal / Extra-curricular ---
  {
    slug: "llm-uncertainty-quantification",
    title: "LLM Uncertainty Quantification (Confidence & Calibration)",
    summary:
      "Chain of LLMs for data annotation with per-model confidence and ECE calibration; CSV uploads, token input, selectable ensembles; React UI + Node/Python orchestration.",
    tags: ["Python", "pandas", "Machine Learning", "Git/GitHub", "Calibration"],
    year: 2025,
    repo: "https://github.com/VarunP3000/ConfidenceScoringProject",
    live: "",
    problem:
      "Teams need a reliable way to aggregate multiple LLM judgments while exposing model-level confidence and a calibrated ensemble score for downstream decisions.",
    approach: [
      "Built upload flow for CSV prompts + HF token; UI lets users pick N models and set per-model thresholds.",
      "Node (Express) orchestrates calls; Python workers compute per-model confidence and Expected Calibration Error (ECE).",
      "Ensemble combines votes + confidences; exported JSON/CSV with per-row rationale hooks.",
    ],
    results: [
      "Reduced single-model volatility with a simple confidence-weighted ensemble.",
      "ECE reporting surfaced over-/under-confident models, guiding which to include.",
      "UI enabled quick ablations (add/drop models, tweak thresholds) without code changes.",
    ],
    limitations:
      "Latency grows with the number of models/API calls; judgments can still correlate if models share training data.",
    nextSteps: [
      "Add temperature/seed sweeps and diversity metrics; plug in cost tracking.",
      "Support abstain/triage policy and per-task calibration datasets.",
    ],
  },
  {
    slug: "stock-market-cpi-to-sp500-fullstack",
    title: "Stock Market Prediction — Full-Stack (CPI → S&P 500)",
    summary:
      "FastAPI backend for regressions/ensembles with backtesting + quantiles; Next.js UI for scenario runs and adjustable model weights; studies inflation lag features vs returns.",
    tags: ["Python", "scikit-learn", "pandas", "Machine Learning", "Git/GitHub", "FastAPI", "Next.js"],
    year: 2025,
    repo: "https://github.com/VarunP3000/stock-cost-of-living-app",
    live: "",
    problem:
      "Can CPI components and lags explain or forecast near-term S&P 500 movements well enough to inform simple allocation rules?",
    approach: [
      "Ingested CPI categories + S&P returns; engineered rolling lags/diffs; aligned on business days.",
      "Baselines: Ridge/ElasticNet/GBR; time-series splits and walk-forward validation; simple ensemble.",
      "Backtesting endpoints return RMSE/MAE, coverage for quantiles, and diagnostics for leakage.",
    ],
    results: [
      "ElasticNet baseline provided strong bias-variance balance; inflation lags showed consistent signal windows.",
      "Dashboard allowed what-if scenarios; quantile bands helped visualize tail risk.",
    ],
    limitations:
      "Macroeconomic regimes shift; limited stationarity and small sample for certain CPI sub-series.",
    nextSteps: [
      "Add macro features (rates, term spread); try XGBoost/LightGBM; expand evaluation to directional metrics.",
      "Deploy caching for repeated scenarios.",
    ],
  },

  // --- CSE 373 — DS&A (Java) ---
  {
    slug: "cse373-deques-arraydeque-linkeddeque",
    title: "Deques (ArrayDeque & LinkedDeque)",
    summary:
      "Circular-array deque with wrap indices + amortized resize; sentinel DLL with Θ(1) end ops; fixed wrap/resize bug; benchmarks vs ArrayList-based deque.",
    tags: ["Java", "Data Structures", "JUnit/Testing", "Performance Optimization"],
    year: 2024,
    repo: "https://github.com/VarunP3000/DataStructures-Algorithms/tree/main/src/main/java/deques",
    problem:
      "Provide double-ended operations with Θ(1) amortized adds/removes without the shifting penalty of array lists.",
    approach: [
      "ArrayDeque: ring buffer + modular indices; careful resize preserving order.",
      "LinkedDeque: sentinel DLL to avoid null checks; tight O(1) end ops.",
      "Unit tests for edge cases (wraparound, empty/full transitions); micro-benchmarks.",
    ],
    results: [
      "ArrayDeque removeFirst beats ArrayList.remove(0) by avoiding Θ(n) shifts.",
      "Bug fix: corrected wrap+grow ordering to preserve logical sequence.",
    ],
    limitations: "Iterator invalidation rules and concurrent mutation are out of scope.",
    nextSteps: "Add fail-fast iterator and capacity hints; compare against java.util.ArrayDeque.",
  },
  {
    slug: "cse373-autocomplete-4-data-structures",
    title: "Autocomplete (4 Data Structures)",
    summary:
      "Baselines: sequential scan + binary-search window. Implemented TST with end markers + DFS collector. Compared against TreeSet; adversarial tests and asymptotic analysis.",
    tags: ["Java", "Data Structures", "Algorithms", "JUnit/Testing"],
    year: 2024,
    repo: "https://github.com/VarunP3000/DataStructures-Algorithms/tree/main/src/main/java/autocomplete",
    problem: "Find fast prefix matches across dictionary sizes with predictable memory/time trade-offs.",
    approach: [
      "Sequential scan and sorted-array binary-window as baselines.",
      "TST with path compression and terminal flags; DFS to collect k suggestions.",
      "Measured query latency vs memory vs baseline TreeSet ceiling.",
    ],
    results: [
      "TST dominated at medium/large vocabularies for short prefixes.",
      "Sorted-array window strong when memory is tight and prefixes are long.",
    ],
    limitations: "No fuzzy matching or unicode normalization in this version.",
    nextSteps: "Add top-k with frequency priors; memory-lean DAWG variant.",
  },
  {
    slug: "cse373-priority-queues-minpq",
    title: "Priority Queues (MinPQ)",
    summary:
      "UnsortedArrayMinPQ (linear membership/changePriority) → HeapMinPQ (remove+reinsert) → optimized binary heap + index map for near Θ(log n) changePriority; benchmarks vs DoubleMapMinPQ.",
    tags: ["Java", "Data Structures", "Algorithms", "Performance Optimization", "JUnit/Testing"],
    year: 2024,
    repo: "https://github.com/VarunP3000/DataStructures-Algorithms/tree/main/src/main/java/minpq",
    problem: "Support decrease-key efficiently for graph algorithms (e.g., Dijkstra).",
    approach: [
      "Baseline unsorted array with simple scans.",
      "Binary heap with index map for O(log n) changePriority.",
      "Stress tests on adversarial change sequences; micro-benchmarks.",
    ],
    results: ["Index-mapped heap maintained heap invariants and outperformed baseline on changePriority-heavy workloads."],
    limitations: "No meld operation; not a pairing/fibonacci heap.",
    nextSteps: "Add decrease-key API parity and optional stable tie-breaking.",
  },
  {
    slug: "cse373-shortest-paths-and-seam-carving",
    title: "Shortest Paths & Seam Carving",
    summary:
      "Two seam graphs: materialized adjacency list and on-demand generative; solvers: DAG topo-relax, Dijkstra; also Bellman-Ford/SPFA/A*; added DP seam finder and runtime comparisons.",
    tags: ["Java", "Graphs", "Dynamic Programming", "Shortest Paths", "Algorithms"],
    year: 2024,
    repo: "https://github.com/VarunP3000/DataStructures-Algorithms/tree/main/src/main/java/graphs",
    problem: "Find minimal-energy image seams efficiently under multiple graph formulations.",
    approach: [
      "Energy model + graph encodings (materialized vs generative).",
      "Compared Dijkstra/topo-relax vs pure DP on acyclic seam graphs.",
      "Benchmarked across resolutions and energy distributions.",
    ],
    results: ["DP outperformed graph-based methods on acyclic seam structures; generative graphs reduced memory footprint."],
    limitations: "No multiseam/parallel extraction; energy model fixed.",
    nextSteps: "Add forward energy and batch seam removal with caching.",
  },

  // --- CSE/STAT 416 — AI/ML (Python) ---
  {
    slug: "c416-house-prices",
    title: "House Prices Prediction",
    summary:
      "Imputation/encoding with robust splits; linear + regularized baselines; validation-curve sweeps; RMSE monitoring and coefficient interpretation.",
    tags: ["Python", "scikit-learn", "pandas", "Machine Learning"],
    year: 2025,
    repo: "https://github.com/VarunP3000/PythonAI-MLProjects/blob/main/hw0.ipynb",
    problem: "Predict home prices from tabular features while avoiding leakage.",
    approach: [
      "Pipelines for imputation/standardization; proper train/val/test.",
      "Linear, Ridge, LASSO; hyperparameter sweeps via validation curves.",
    ],
    results: ["Regularized models reduced variance; interpretable coefficients highlighted dominant drivers."],
    limitations: "Limited feature crafting; no tree ensembles in baseline.",
    nextSteps: "Try GBMs/XGBoost and SHAP for explanations.",
  },
  {
    slug: "c416-ridge-and-lasso",
    title: "Ridge & LASSO Regression",
    summary:
      "Engineered polynomial/√ features; standardized inside pipelines; swept α; contrasted Ridge vs LASSO; coefficient-path visualizations.",
    tags: ["Python", "scikit-learn", "Machine Learning"],
    year: 2025,
    repo: "https://github.com/VarunP3000/PythonAI-MLProjects/blob/main/Ridge%26Lasso.ipynb",
    problem: "Trade off bias/variance and feature selection under collinearity.",
    approach: ["Feature engineering + scaling pipelines", "Grid over α; path plots for interpretability"],
    results: ["LASSO produced sparse models; Ridge stabilized coefficients with slightly better generalization on this dataset."],
    limitations: "Linear hypothesis; interaction space limited.",
    nextSteps: "ElasticNet sweep; feature grouping and stability selection.",
  },
  {
    slug: "c416-sentiment-analysis-amazon",
    title: "Sentiment Analysis (Amazon Reviews)",
    summary:
      "CountVectorizer features; train/val/test with majority baseline; logistic regression + confusion matrix; L2 regularization and term analysis.",
    tags: ["Python", "scikit-learn", "Machine Learning"],
    year: 2025,
    repo: "https://github.com/VarunP3000/PythonAI-MLProjects/blob/main/SentimentAnalysisWithLogisticRegression.ipynb",
    problem: "Classify review sentiment with simple, explainable models.",
    approach: ["BoW with unigrams/bigrams; C sweep; evaluation on held-out set."],
    results: ["Logistic regression beat baseline substantially; top positive/negative terms aligned with intuition."],
    limitations: "No handling of negation scope or context; no transformer embeddings.",
    nextSteps: "Try TF-IDF + linear SVM; DistilBERT for comparison.",
  },
  {
    slug: "c416-loan-default-trees-rf",
    title: "Loan Default Risk (Trees & Custom Random Forest)",
    summary:
      "One-hot encoding + class imbalance handling; tuned depth/min samples; custom RF with bootstrap; learning curves and importances.",
    tags: ["Python", "scikit-learn", "Machine Learning"],
    year: 2025,
    repo: "https://github.com/VarunP3000/PythonAI-MLProjects/blob/main/LoanSafety.ipynb",
    problem: "Predict loan default with non-linear interactions and imbalanced labels.",
    approach: ["Decision trees with grid search; custom RF (bagging) with OOB-style evaluation."],
    results: ["RF reduced variance vs single tree; most important features matched domain expectations."],
    limitations: "No calibration step; threshold fixed at 0.5.",
    nextSteps: "Add PR-AUC optimization and probability calibration.",
  },
  {
    slug: "c416-cifar10-nets-a-b-c-d",
    title: "CIFAR-10 Image Classification (Net A/B/C/D)",
    summary:
      "PyTorch MLP→CNN stacks; GPU training, early stopping, LR scheduling; met target accuracy; misclassification analysis.",
    tags: ["Python", "PyTorch", "Deep Learning", "Neural Networks"],
    year: 2025,
    repo: "https://github.com/VarunP3000/PythonAI-MLProjects/blob/main/DeepLearning.ipynb",
    problem: "Reach target validation accuracy on CIFAR-10 with compact architectures.",
    approach: ["Progressive depth; data augmentation; schedulers + early stop.", "Tracked per-epoch metrics; confusion matrix."],
    results: ["Target accuracy achieved; error clusters revealed confusable classes (cat/dog, ship/airplane edge cases)."],
    limitations: "Limited capacity; no modern regularizers or cutmix/mixup.",
    nextSteps: "Add residual blocks and stronger aug; compare to ViT-tiny.",
  },
  {
    slug: "c416-kmeans-from-scratch-wikipedia-tfidf",
    title: "K-Means from Scratch (Wikipedia, TF-IDF)",
    summary:
      "Scratch k-means with inertia & convergence; k-means++ init + multi-restart; elbow/heterogeneity; TF-IDF clustering with topic inspection.",
    tags: ["Python", "NumPy", "Clustering", "Machine Learning"],
    year: 2025,
    repo: "https://github.com/VarunP3000/PythonAI-MLProjects/blob/main/KMeansWithTextData.ipynb",
    problem: "Explore unsupervised structure and stability under random initializations.",
    approach: ["Vectorized implementation; ++ init; multiple restarts; TF-IDF features; elbow heuristic."],
    results: ["Stable clusters with ++ and restarts; top-term inspection produced coherent topics."],
    limitations: "Requires spherical clusters; k must be chosen; sensitive to scaling.",
    nextSteps: "Try K-Medoids and spectral clustering; silhouette search for k.",
  },
  {
    slug: "c416-twitter-topic-modeling-nmf",
    title: "Twitter Topic Modeling (NMF)",
    summary:
      "TF-IDF preprocessing (stopwords/case/punct); NMF with k=5 and k=3; top words per topic; tweet→topic weights; dominant topic assignment.",
    tags: ["Python", "NMF/Topic Modeling", "Machine Learning"],
    year: 2025,
    repo: "https://github.com/VarunP3000/PythonAI-MLProjects/blob/main/RecommendationWithText.ipynb",
    problem: "Discover latent themes in short texts without labels.",
    approach: ["Preprocess to TF-IDF; factorize with NMF; inspect components and document loadings."],
    results: ["Topics were interpretable; flagged an outlier cluster driven by rare terms."],
    limitations: "Short texts are sparse; topic drift across runs.",
    nextSteps: "Stabilize via anchoring/seeded NMF; compare to LDA.",
  },

  // --- INFO 201 — Data Science & Informatics Foundations (R) ---
  {
    slug: "info201-ps01-basic-r",
    title: "Basic R (variables, logic, strings, loops, functions)",
    summary:
      "Small R programs for derived quantities, boolean logic, strings, and basic functions; stringr templating; classic loops; formatted outputs and checks.",
    tags: ["R", "tidyverse"],
    year: 2024,
    repo: "https://github.com/VarunP3000/RDataScienceProjects/blob/main/Info201Assignments/INFO201ps01.R",
    problem: "Build fluency in R control flow and string processing.",
    approach: ["Practice tasks across numbers/strings/loops; lightweight validation prints."],
    results: ["Correct outputs and idiomatic stringr usage."],
    limitations: "Intro scale; no vectorization yet.",
    nextSteps: "Refactor with vectorized ops where possible.",
  },
  {
    slug: "info201-ps02-vectors",
    title: "Vectors, Vectorization & Lists",
    summary:
      "Vector creation/slicing and fully vectorized transforms; award calculator with named vectors + logical indexing; lists for structured data; dice simulator.",
    tags: ["R"],
    year: 2024,
    repo: "https://github.com/VarunP3000/RDataScienceProjects/blob/main/Info201Assignments/INFO201ps02.R",
    problem: "Leverage vectorization for speed and clarity.",
    approach: ["Use named vectors and logical masks; move loops to vector ops."],
    results: ["Cleaner, faster solutions; simpler code paths."],
    limitations: "Small examples; no large perf profiling.",
    nextSteps: "Benchmark vs loop equivalents; micro-optimize where useful.",
  },
  {
    slug: "info201-ps03-rmarkdown-filesystems",
    title: "R Markdown, Filesystems & Control Flow",
    summary:
      "Reproducible R Markdown mixing narrative, code, and images; paths, file enumeration; concise ifelse pipelines; inline code.",
    tags: ["R"],
    year: 2024,
    repo: "https://github.com/VarunP3000/RDataScienceProjects/blob/main/Info201Assignments/INFO201ps03.Rmd",
    problem: "Make results reproducible and portable.",
    approach: ["Parameterize paths; keep code chunks minimal; inline key values."],
    results: ["Self-contained report builds cleanly."],
    limitations: "Local FS only; no remote data pulls.",
    nextSteps: "Add renv and data-versioning hooks.",
  },
  {
    slug: "info201-ps04-life-expectancy-nyc-flights",
    title: "Life Expectancy & NYC Flights",
    summary:
      "Counted missings, per-country growth, regional summaries; labeled scatterplots; mean-diff equivalence; NYC flights cleaning and SEA delay analysis.",
    tags: ["R", "tidyverse"],
    year: 2024,
    repo: "https://github.com/VarunP3000/RDataScienceProjects/blob/main/Info201Assignments/INFO201ps04.Rmd",
    problem: "Explore relationships and communicate clearly with plots.",
    approach: ["Tidy verbs for transforms; labeled ggplots; grouped summaries."],
    results: ["Readable visuals; verified statistical identities."],
    limitations: "No causal claims; limited covariates.",
    nextSteps: "Facets over time; add uncertainty ribbons.",
  },
  {
    slug: "info201-ps05-co2-gapminder",
    title: "CO2, Gapminder & Fertility vs Life Expectancy",
    summary:
      "Audited ISO codes; total vs per-capita CO₂; regional means (1960/2016); orange-tree growth; continent and country trend plots.",
    tags: ["R", "tidyverse"],
    year: 2024,
    repo: "https://github.com/VarunP3000/RDataScienceProjects/blob/main/Info201Assignments/INFO201ps05.html",
    problem: "Compare carbon metrics across regions and time cleanly.",
    approach: ["Standardize identifiers; tidy pivots; grouped summaries + plots."],
    results: ["Consistent identifiers; clear temporal contrasts."],
    limitations: "No adjustments for trade/embedded emissions.",
    nextSteps: "Normalize by GDP; add consumption-based CO₂ if available.",
  },
  {
    slug: "info201-ps06-co2-vs-temperature",
    title: "CO2 vs Temperature (Scripps, HadCRUT, UAH)",
    summary:
      "Cleaned monthly CO₂; built continuous time axis; baseline anomalies; yearly aggregation; merged surface & satellite; decade-colored trends.",
    tags: ["R", "tidyverse"],
    year: 2024,
    repo: "https://github.com/VarunP3000/RDataScienceProjects/blob/main/Info201Assignments/INFO201ps06.Rmd",
    problem: "Visualize CO₂ rise alongside temperature anomalies.",
    approach: ["QC monthly series; align baselines; merge datasets; plot trends."],
    results: ["Co-movement visible; clean anomaly framing."],
    limitations: "No causal inference; possible coverage biases.",
    nextSteps: "Lag analysis; compare ENSO phases.",
  },
  {
    slug: "info201-infolab5-dataframes-subsetting",
    title: "Data Frames, Subsetting & Simple Analytics",
    summary:
      "Seahawks tidyframe; margin/win indicators; HR salaries raises; vectorized filters; tidy printing.",
    tags: ["R"],
    year: 2024,
    repo: "https://github.com/VarunP3000/RDataScienceProjects/blob/main/InfoLab5.Rmd",
    problem: "Practice tidy data and quick EDA patterns.",
    approach: ["Vectorized transforms; grouped filters; succinct outputs."],
    results: ["Concise analyses with minimal boilerplate."],
    limitations: "Toy-scale; not production reporting.",
    nextSteps: "Package helper functions for reuse.",
  },
  {
    slug: "info201-capstone-stock-retirement",
    title: "Capstone — Stock Market Retirement (Repo)",
    summary:
      "Reproducible R project for retirement-style stock analysis; tidy ingestion/transforms; labeled plots; transparent assumptions; organized repo.",
    tags: ["R", "tidyverse"],
    year: 2024,
    repo: "https://github.com/mojipao/Stock-Market-Retirement",
    problem: "Evaluate retirement-style strategies with reproducible workflows.",
    approach: ["Structured R project; parameterized analysis; tidy plots with clear labels."],
    results: ["Reviewable repo with replicable figures."],
    limitations: "Backtest bias possible; limited asset classes.",
    nextSteps: "Add bond/commodity sleeves and drawdown metrics.",
  },

  // --- STAT/BIOSTAT 534 — Statistical Computing (UW) ---
  {
    slug: "stat534-hw1-bayesian-linear-models",
    title: "Bayesian Linear Models: Log-Determinant & Marginal Likelihood",
    summary:
      "Eigendecomposition log-det; closed-form marginal likelihood for conjugate normal LMs; arbitrary subset selection; verified on ER dataset.",
    tags: ["R", "Statistical Computing"],
    year: 2025,
    repo: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/Stat534HW1",
    problem: "Compute log-dets and evidences robustly for model comparison.",
    approach: ["Center/standardize; eigendecomp for stability; closed-form evidence evaluation over subsets."],
    results: ["Matches textbook values; numerically stable across subsets."],
    limitations: "Conjugate priors only; small-n datasets.",
    nextSteps: "Add ridge priors and WAIC/LOO comparison.",
  },
  {
    slug: "stat534-hw2-greedy-logistic-selection",
    title: "Greedy Variable Selection for Logistic Regression",
    summary:
      "AIC/BIC helpers; robust glm wrappers; forward/backward greedy AIC with skip on non-convergence; compact models align with BIC.",
    tags: ["R", "Statistical Computing"],
    year: 2025,
    repo: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/Stat534HW2",
    problem: "Select parsimonious logistic models efficiently.",
    approach: ["Greedy add/remove; guard rails for glm failures; AIC/BIC tracking."],
    results: ["Forward AIC converged to compact models; close to BIC picks on dataset tested."],
    limitations: "Greedy may miss global optimum; correlation sensitive.",
    nextSteps: "Compare with L1 logistic and stepwise with interactions.",
  },
  {
    slug: "stat534-hw3-mc3-model-selection",
    title: "Stochastic Model Selection via MC3",
    summary:
      "MC3 over add/remove-one neighborhoods; rcdd linearity checks; MH with neighbor-count correction; compared to greedy selections.",
    tags: ["R", "Statistical Computing", "Bayesian Methods"],
    year: 2025,
    repo: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/Stat534HW3",
    problem: "Explore model space without exhaustive enumeration.",
    approach: ["MC3 proposals; acceptance with neighborhood correction; chains diagnostics; rcdd validity filters."],
    results: ["Stochastic search recovered high-scoring subsets similar to greedy but with alternatives for correlated features."],
    limitations: "Chain mixing depends on proposal tuning.",
    nextSteps: "Parallel tempering and adaptive proposals.",
  },
  {
    slug: "stat534-hw4-laplace-mh-logistic",
    title: "Bayesian Univariate Logistic Regression (Laplace + MH)",
    summary:
      "Posterior mode via Newton–Raphson; Laplace approximation; MH sampler; parallelized 60 fits with snow; posterior means + MLE sanity checks.",
    tags: ["R", "Statistical Computing", "Bayesian Methods", "Parallel Computing"],
    year: 2025,
    repo: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/hwk4",
    problem: "Approximate posteriors quickly and validate with sampling.",
    approach: ["Mode finding; Laplace evidence; MH initialized at mode; log-accept tracking; parallel runs."],
    results: ["Laplace close to MH posterior means; acceptance stable with tuned proposals."],
    limitations: "Univariate only; Gaussian priors.",
    nextSteps: "Extend to multivariate GLMs; adaptive MH.",
  },
  {
    slug: "stat534-hw5-marginal-likelihood-cpp",
    title: "Marginal Likelihood for Linear Regression (C/C++: LAPACK & GSL)",
    summary:
      "Two high-perf C/C++ versions (LAPACKE, GSL) for LM marginal likelihood; GEMM/solve/log-det with careful memory/layout; matched R baseline and spec.",
    tags: ["C/C++", "LAPACK", "GSL", "Statistical Computing", "Performance Optimization"],
    year: 2025,
    repo: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/hwk5",
    problem: "Compute evidences fast and accurately in native code.",
    approach: ["Linear algebra kernels via BLAS/LAPACK & GSL; attention to row/col major; tests vs R."],
    results: ["Numeric parity with R; strong runtime characteristics on medium-n."],
    limitations: "CPU-only; no batching across many models.",
    nextSteps: "Vectorize across models; add OpenMP.",
  },
  {
    slug: "stat534-hw6-recursive-det-topk",
    title: "Recursive Determinant & Top-K Regression Search",
    summary:
      "Pure recursive determinant with exact base cases; safe memory; bounded top-K list; enumerated ≤2-predictor models.",
    tags: ["C/C++", "LAPACK", "Algorithms", "Statistical Computing"],
    year: 2025,
    repo: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/hwk6",
    problem: "Teach algorithmic fundamentals and ranking structures for model search.",
    approach: ["Laplace expansion recursion; singly linked list for top-K with dedupe; write best models to disk."],
    results: ["Correct determinants and consistent top-K ranking.",
    ],
    limitations: "Exponential growth beyond tiny k; pedagogical emphasis.",
    nextSteps: "Heuristics/pruning; branch-and-bound.",
  },
  {
    slug: "stat534-hw7-mvn-covariance",
    title: "Multivariate Normal Sampling & Covariance Estimation",
    summary:
      "Empirical covariance; Cholesky sampling; BLAS transforms; 10k sims; sample covariance matches target.",
    tags: ["C/C++", "GSL", "Statistical Computing"],
    year: 2025,
    repo: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/hwk7",
    problem: "Validate sampling pipeline against theoretical covariance.",
    approach: ["Compute Σ̂; factor with Cholesky; generate Z and transform; compare Σ̂ to Σ."],
    results: ["Element-wise closeness within tolerance over 10k draws."],
    limitations: "Gaussian-only; no sparse structures.",
    nextSteps: "Try low-rank updates and block sampling.",
  },
  {
    slug: "stat534-final-mpi-volleyball",
    title: "MPI Volleyball Match Simulator",
    summary:
      "13-process simulation (referee + 12 players) with point-to-point messages; rally probabilities; compact payloads; scoring/set logic; clean termination.",
    tags: ["C/C++, MPI", "Parallel Computing"],
    year: 2025,
    repo: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/Stat534Final",
    problem: "Model distributed decision flow with explicit message passing.",
    approach: ["MPI ranks for roles; GSL RNG; serve/rally probabilities; referee tracks scoring and sets; termination messages."],
    results: ["Deterministic logs and correct match rules; portable build with mpic++."],
    limitations: "No network failures; simplified rally model.",
    nextSteps: "Add stochastic latency and visualization.",
  },

  // --- Stat 311 — R Labs ---
  {
    slug: "rstat-lab2-nyc-flights-delays",
    title: "NYC Flight Delays — Exploratory Analysis (R Lab 2)",
    summary:
      "Tidyverse EDA on NYC flights; histogram binning, route filters; Feb-SFO subset; ranked months by delay to show seasonality.",
    tags: ["R", "tidyverse", "ggplot2", "dplyr", "Data Wrangling"],
    year: 2025,
    repo: "https://github.com/VarunP3000/RStatisticalComputing/blob/main/RStatisticalComputing/StatsLab2.pdf",
    problem: "Surface delay patterns across routes and months.",
    approach: ["Histogram tuning; route filtering; grouped summaries; seasonal ranking."],
    results: ["Clear seasonal delay structure; route-specific insights."],
    limitations: "No weather/ATC covariates.",
    nextSteps: "Merge NOAA weather and holiday calendars.",
  },
  {
    slug: "rstat-lab3-random-simulations",
    title: "Random Simulations — Coins & Dice (R Lab 3)",
    summary:
      "Sampling with/without replacement; empirical proportions/variance; manual vs built-ins; more trials show convergence.",
    tags: ["R", "Base R", "Probability and Statistics"],
    year: 2025,
    repo: "https://github.com/VarunP3000/RStatisticalComputing/blob/main/RStatisticalComputing/StatsLab3.pdf",
    problem: "Demonstrate LLN behavior and variance formulas via simulation.",
    approach: ["Generate samples; compute stats manually and with R; increase N; compare."],
    results: ["Convergence toward theoretical variance as N↑."],
    limitations: "Toy distributions only.",
    nextSteps: "Add CI coverage experiments.",
  },
  {
    slug: "rstat-lab4-hot-hand-fast-food",
    title: "Hot Hand & Fast-Food Normality (R Lab 4)",
    summary:
      "Kobe streak analysis vs independent shooter; fast-food calories from fat with normal overlays and QQ checks.",
    tags: ["R", "tidyverse", "ggplot2", "Probability and Statistics"],
    year: 2025,
    repo: "https://github.com/VarunP3000/RStatisticalComputing/blob/main/RStatisticalComputing/StatsLab4.pdf",
    problem: "Test hot-hand hypothesis and examine normality assumptions.",
    approach: ["calc_streak and simulated baseline; QQ/qqnormsim for normality; group comparisons."],
    results: ["No hot-hand evidence; McD vs DQ distribution differences visualized.",
    ],
    limitations: "Simple independence model may miss tempo effects.",
    nextSteps: "Markov shooter model; robust stats for skew.",
  },
  {
    slug: "rstat-lab5-sampling-distributions",
    title: "Sampling Distributions & Population Proportions (R Lab 5)",
    summary:
      "15k resamples via infer + tidyverse; p-hat experiment; compare n=10/50/100 for normality and SE.",
    tags: ["R", "tidyverse", "Probability and Statistics"],
    year: 2025,
    repo: "https://github.com/VarunP3000/RStatisticalComputing/blob/main/RStatisticalComputing/StatsLab5.pdf",
    problem: "Visualize how sample size affects sampling variability.",
    approach: ["Repeat sampling; histogram sampling distributions; compute SE across n."],
    results: ["As n↑, distributions normalize; mean→truth; SE↓ as expected."],
    limitations: "Synthetic setup; fixed p.",
    nextSteps: "Vary p and add CI coverage plots.",
  },
  {
    slug: "rstat-lab6-texting-and-driving",
    title: "Texting & Driving — Hypothesis Testing and Confidence Intervals (R Lab 6)",
    summary:
      "YRBSS texting-while-driving proportion; 99%/95% CIs; MOE vs p; test H0: p=0.05 (p≈0.00014).",
    tags: ["R", "tidyverse", "Probability and Statistics"],
    year: 2025,
    repo: "https://github.com/VarunP3000/RStatisticalComputing/blob/main/RStatisticalComputing/StatsLab6.pdf",
    problem: "Estimate prevalence and test policy-relevant hypotheses.",
    approach: ["Compute CIs; visualize MOE; run hypothesis test and interpret."],
    results: ["Strong evidence true p exceeds 5% threshold."],
    limitations: "Survey bias possible; self-reporting error.",
    nextSteps: "Weighting adjustments and subgroup analysis.",
  },
  {
    slug: "rstat-lab7-weight-vs-activity",
    title: "Weight vs Physical Activity — Two-Sample Inference (R Lab 7)",
    summary:
      "YRBSS active vs inactive groups; boxplots; assumptions checked; two-sample t-tests; p≈0.0002/0.0001.",
    tags: ["R", "tidyverse", "Probability and Statistics"],
    year: 2025,
    repo: "https://github.com/VarunP3000/RStatisticalComputing/blob/main/RStatisticalComputing/StatsLab7.pdf",
    problem: "Compare group means under standard conditions.",
    approach: ["EDA + diagnostics; two-sample tests; effect size reporting."],
    results: ["Statistically significant differences in means."],
    limitations: "Observational data; confounders uncontrolled.",
    nextSteps: "Regression with controls; propensity matching.",
  },
  {
    slug: "rstat-lab8-linear-regression-freedom",
    title: "Linear Regression — Personal Freedom vs Expression Control (R Lab 8)",
    summary:
      "Fit pf_score ~ pf_expression_control; slope 0.54, intercept 4.28; diagnostics confirmed model validity.",
    tags: ["R", "tidyverse", "ggplot2", "Probability and Statistics"],
    year: 2025,
    repo: "https://github.com/VarunP3000/RStatisticalComputing/blob/main/RStatisticalComputing/StatsLab8.pdf",
    problem: "Quantify association between freedom and expression control indices.",
    approach: ["Fit simple OLS; check residual plots and normality; report coefficients.",
    ],
    results: ["Positive slope with good diagnostics; interpretable linear fit."],
    limitations: "Bivariate only; omitted variables possible.",
    nextSteps: "Multiple regression and robustness checks.",
  },
];
