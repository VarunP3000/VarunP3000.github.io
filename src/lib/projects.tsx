import type { Project } from "../components/ProjectCard";

// Helper: choose a readable slug per project
const s = (x: string) =>
  x
    .toLowerCase()
    .replace(/&/g, "and")
    .replace(/→/g, "-")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "");

export const projects: Project[] = [
  // Personal / Extra-curricular
  {
    slug: "llm-uncertainty-quantification",
    title: "LLM Uncertainty Quantification (Confidence & Calibration)",
    summary:
      "Chain of LLMs for data annotation with per-model confidence and ECE calibration; CSV uploads, token input, and selectable ensemble models across a React UI; Node.js + Python backend for workflow orchestration.",
    tags: ["Python", "pandas", "Machine Learning", "Git/GitHub", "Calibration"],
    year: 2025,
    repo: "https://github.com/VarunP3000/ConfidenceScoringProject",
    live: "",
  },
  {
    slug: "stock-market-cpi-to-sp500-fullstack",
    title: "Stock Market Prediction — Full-Stack (CPI → S&P 500)",
    summary:
      "FastAPI backend for regression/ensembles with backtesting and quantile outputs; Next.js front-end to run scenarios and adjust model weights; clean API contracts, CORS-safe deployment, readable metrics/plots.",
    tags: ["Python", "scikit-learn", "pandas", "Machine Learning", "Git/GitHub", "FastAPI", "Next.js"],
    year: 2025,
    repo: "https://github.com/VarunP3000/stock-cost-of-living-app",
    live: "",
  },

  // CSE 373 — Data Structures & Algorithms (Java)
  {
    slug: "cse373-deques-arraydeque-linkeddeque",
    title: "Deques (ArrayDeque & LinkedDeque)",
    summary:
      "Circular-array deque with wrap indices + amortized resize; sentinel DLL with Θ(1) end ops; fixed wrap/resize bug; benchmarks vs ArrayList-backed deque; tests explaining why ArrayList.remove(0) is Θ(n).",
    tags: ["Java", "Data Structures", "JUnit/Testing", "Performance Optimization"],
    year: 2024,
    repo: "https://github.com/VarunP3000/DataStructures-Algorithms/tree/main/src/main/java/deques",
  },
  {
    slug: "cse373-autocomplete-4-data-structures",
    title: "Autocomplete (4 Data Structures)",
    summary:
      "Baselines: sequential scan + binary-search window. Implemented TST with end-markers and DFS collector. Compared against TreeSet; unit tests for adversarial cases; analyzed worst- vs average-case lookups.",
    tags: ["Java", "Data Structures", "Algorithms", "JUnit/Testing"],
    year: 2024,
    repo: "https://github.com/VarunP3000/DataStructures-Algorithms/tree/main/src/main/java/autocomplete",
  },
  {
    slug: "cse373-priority-queues-minpq",
    title: "Priority Queues (MinPQ)",
    summary:
      "UnsortedArrayMinPQ (linear membership/changePriority); HeapMinPQ via PriorityQueue (remove+reinsert); optimized binary heap + index map for near Θ(log n) changePriority; benchmarked vs DoubleMapMinPQ.",
    tags: ["Java", "Data Structures", "Algorithms", "Performance Optimization", "JUnit/Testing"],
    year: 2024,
    repo: "https://github.com/VarunP3000/DataStructures-Algorithms/tree/main/src/main/java/minpq",
  },
  {
    slug: "cse373-shortest-paths-and-seam-carving",
    title: "Shortest Paths & Seam Carving",
    summary:
      "Seam graphs: materialized adjacency list and on-demand generative. Solvers: DAG topological (relax), Dijkstra; also used Bellman-Ford, SPFA, A*. Added DP seam finder and compared runtime/allocations.",
    tags: ["Java", "Graphs", "Dynamic Programming", "Shortest Paths", "Algorithms"],
    year: 2024,
    repo: "https://github.com/VarunP3000/DataStructures-Algorithms/tree/main/src/main/java/graphs",
  },

  // CSE/STAT 416 — AI/ML (Python)
  {
    slug: "c416-house-prices",
    title: "House Prices Prediction",
    summary:
      "Imputation/encoding with robust train/val/test splits. Linear + regularized baselines; hyperparameter sweeps via validation curves; tracked RMSE/generalization and interpreted coefficient effects.",
    tags: ["Python", "scikit-learn", "pandas", "Machine Learning"],
    year: 2025,
    repo: "https://github.com/VarunP3000/PythonAI-MLProjects/blob/main/hw0.ipynb",
  },
  {
    slug: "c416-ridge-and-lasso",
    title: "Ridge & LASSO Regression",
    summary:
      "Engineered polynomial/√ features; standardized via pipelines; swept α (λ) values; contrasted Ridge shrinkage vs LASSO sparsity on held-out RMSE; coefficient-path visualizations for bias–variance trade-offs.",
    tags: ["Python", "scikit-learn", "Machine Learning"],
    year: 2025,
    repo: "https://github.com/VarunP3000/PythonAI-MLProjects/blob/main/Ridge%26Lasso.ipynb",
  },
  {
    slug: "c416-sentiment-analysis-amazon",
    title: "Sentiment Analysis (Amazon Reviews)",
    summary:
      "CountVectorizer features; train/val/test split with majority baseline; logistic regression + accuracy, confusion matrix, class probabilities; L2 regularization and top positive/negative term analysis.",
    tags: ["Python", "scikit-learn", "Machine Learning"],
    year: 2025,
    repo: "https://github.com/VarunP3000/PythonAI-MLProjects/blob/main/SentimentAnalysisWithLogisticRegression.ipynb",
  },
  {
    slug: "c416-loan-default-trees-rf",
    title: "Loan Default Risk (Trees & Custom Random Forest)",
    summary:
      "One-hot encoding + class imbalance handling; tuned depth/min samples via GridSearchCV; custom RF with bootstrapping; learning curves & feature importances; compared high-variance tree vs forest.",
    tags: ["Python", "scikit-learn", "Machine Learning"],
    year: 2025,
    repo: "https://github.com/VarunP3000/PythonAI-MLProjects/blob/main/LoanSafety.ipynb",
  },
  {
    slug: "c416-cifar10-nets-a-b-c-d",
    title: "CIFAR-10 Image Classification (Net A/B/C/D)",
    summary:
      "PyTorch MLP→CNN stacks with loaders/augmentations; GPU training, early stopping, LR scheduling; tracked accuracy/loss; met target validation accuracy; error analysis on misclassifications.",
    tags: ["Python", "PyTorch", "Deep Learning", "Neural Networks"],
    year: 2025,
    repo: "https://github.com/VarunP3000/PythonAI-MLProjects/blob/main/DeepLearning.ipynb",
  },
  {
    slug: "c416-kmeans-from-scratch-wikipedia-tfidf",
    title: "K-Means from Scratch (Wikipedia, TF-IDF)",
    summary:
      "Scratch k-means with inertia + convergence; k-means++ init and multi-restart stability; elbow/heterogeneity to choose K; clustered TF-IDF; interpreted clusters via top-weight terms/examples.",
    tags: ["Python", "NumPy", "Clustering", "Machine Learning"],
    year: 2025,
    repo: "https://github.com/VarunP3000/PythonAI-MLProjects/blob/main/KMeansWithTextData.ipynb",
  },
  {
    slug: "c416-twitter-topic-modeling-nmf",
    title: "Twitter Topic Modeling (NMF)",
    summary:
      "TF-IDF with stopword/case/punctuation handling; NMF with k=5 and k=3; top words per topic; tweet→topic weights; identified dominant topics and an outlier cluster; summarized themes.",
    tags: ["Python", "NMF/Topic Modeling", "Machine Learning"],
    year: 2025,
    repo: "https://github.com/VarunP3000/PythonAI-MLProjects/blob/main/RecommendationWithText.ipynb",
  },

  // INFO 201 — Data Science & Informatics Foundations (R)
  {
    slug: "info201-ps01-basic-r",
    title: "Basic R (variables, logic, strings, loops, functions)",
    summary:
      "Small R programs for derived quantities, boolean logic, strings, and basic functions; stringr templating/replacements; classic loops (running sums, factorial); formatted outputs and checks.",
    tags: ["R", "tidyverse"],
    year: 2024,
    repo: "https://github.com/VarunP3000/RDataScienceProjects/blob/main/Info201Assignments/INFO201ps01.R",
  },
  {
    slug: "info201-ps02-vectors",
    title: "Vectors, Vectorization & Lists",
    summary:
      "Vector creation/slicing and fully vectorized transforms; award calculator with named vectors + logical indexing; lists for structured data; dice simulator with a test hook.",
    tags: ["R"],
    year: 2024,
    repo: "https://github.com/VarunP3000/RDataScienceProjects/blob/main/Info201Assignments/INFO201ps02.R",
  },
  {
    slug: "info201-ps03-rmarkdown-filesystems",
    title: "R Markdown, Filesystems & Control Flow",
    summary:
      "Reproducible R Markdown mixing narrative, code, and images; working directories/relative paths; file enumeration with list.files/file.info; concise ifelse pipelines; inline code.",
    tags: ["R"],
    year: 2024,
    repo: "https://github.com/VarunP3000/RDataScienceProjects/blob/main/Info201Assignments/INFO201ps03.Rmd",
  },
  {
    slug: "info201-ps04-life-expectancy-nyc-flights",
    title: "Life Expectancy & NYC Flights",
    summary:
      "Counted missings, computed per-country growth, summarized by region; labeled scatterplots; mean-of-diffs vs diff-of-means; NYC flights cleaning, SEA delays, seasonal trends, mph speeds.",
    tags: ["R", "tidyverse"],
    year: 2024,
    repo: "https://github.com/VarunP3000/RDataScienceProjects/blob/main/Info201Assignments/INFO201ps04.Rmd",
  },
  {
    slug: "info201-ps05-co2-gapminder",
    title: "CO2, Gapminder & Fertility vs Life Expectancy",
    summary:
      "Audited ISO codes; contrasted total vs per-capita CO₂ by country; regional means (1960/2016); orange tree growth modeling; continent means and country paths in ggplot2.",
    tags: ["R", "tidyverse"],
    year: 2024,
    repo: "https://github.com/VarunP3000/RDataScienceProjects/blob/main/Info201Assignments/INFO201ps05.html",
  },
  {
    slug: "info201-ps06-co2-vs-temperature",
    title: "CO2 vs Temperature (Scripps, HadCRUT, UAH)",
    summary:
      "Cleaned monthly CO₂; continuous time axis; pre-industrial baseline; temperature anomalies; yearly aggregation and merge with surface/satellite; decade-colored trends.",
    tags: ["R", "tidyverse"],
    year: 2024,
    repo: "https://github.com/VarunP3000/RDataScienceProjects/blob/main/Info201Assignments/INFO201ps06.Rmd",
  },
  {
    slug: "info201-infolab5-dataframes-subsetting",
    title: "Data Frames, Subsetting & Simple Analytics",
    summary:
      "Seahawks results tidyframe (margin, win); targeted subsetting; HR salaries raises and maxima; vectorized arithmetic, logical filters, tidy printing for quick EDA.",
    tags: ["R"],
    year: 2024,
    repo: "https://github.com/VarunP3000/RDataScienceProjects/blob/main/InfoLab5.Rmd",
  },
  {
    slug: "info201-capstone-stock-retirement",
    title: "Capstone — Stock Market Retirement (Repo)",
    summary:
      "Reproducible R project for retirement-style stock analysis; tidy ingestion/transforms; clearly labeled plots; transparent assumptions; organized repo structure.",
    tags: ["R", "tidyverse"],
    year: 2024,
    repo: "https://github.com/mojipao/Stock-Market-Retirement",
  },

  // STAT/BIOSTAT 534 — Statistical Computing (UW)
  {
    slug: "stat534-hw1-bayesian-linear-models",
    title: "Bayesian Linear Models: Log-Determinant & Marginal Likelihood",
    summary:
      "Eigendecomposition-based log-det; closed-form marginal likelihood for conjugate normal LMs; arbitrary predictor subsets on centered/standardized designs; verified on ER dataset.",
    tags: ["R", "Statistical Computing"],
    year: 2025,
    repo: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/Stat534HW1",
  },
  {
    slug: "stat534-hw2-greedy-logistic-selection",
    title: "Greedy Variable Selection for Logistic Regression",
    summary:
      "AIC/BIC helpers; robust glm wrappers; forward/backward greedy AIC with convergence guards; on 60-feature data, forward AIC reached compact models aligning with forward BIC.",
    tags: ["R", "Statistical Computing"],
    year: 2025,
    repo: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/Stat534HW2",
  },
  {
    slug: "stat534-hw3-mc3-model-selection",
    title: "Stochastic Model Selection via MC3",
    summary:
      "MC3 over add/remove-one neighborhoods; rcdd linearity checks to avoid separation; Metropolis–Hastings with neighbor-count correction; compared best subsets/AICs vs HW2.",
    tags: ["R", "Statistical Computing", "Bayesian Methods"],
    year: 2025,
    repo: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/Stat534HW3",
  },
  {
    slug: "stat534-hw4-laplace-mh-logistic",
    title: "Bayesian Univariate Logistic Regression (Laplace + MH)",
    summary:
      "Posterior mode via Newton–Raphson with N(0,1) priors; Laplace evidence; MH sampler from the mode, log-accept tracking; parallelized 60 fits with snow; posterior means + MLE checks.",
    tags: ["R", "Statistical Computing", "Bayesian Methods", "Parallel Computing"],
    year: 2025,
    repo: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/hwk4",
  },
  {
    slug: "stat534-hw5-marginal-likelihood-cpp",
    title: "Marginal Likelihood for Linear Regression (C/C++: LAPACK & GSL)",
    summary:
      "Two high-performance C/C++ implementations (LAPACKE, GSL) for LM marginal likelihood; GEMM, identity-add, solve, log-det with careful memory/layout; matched R baseline and spec.",
    tags: ["C/C++", "LAPACK", "GSL", "Statistical Computing", "Performance Optimization"],
    year: 2025,
    repo: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/hwk5",
  },
  {
    slug: "stat534-hw6-recursive-det-topk",
    title: "Recursive Determinant & Top-K Regression Search",
    summary:
      "Pure recursive determinant via Laplace expansion with exact base cases; memory-safe implementation; bounded top-K list with de-duplication; enumerated ≤2-predictor models.",
    tags: ["C/C++", "LAPACK", "Algorithms", "Statistical Computing"],
    year: 2025,
    repo: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/hwk6",
  },
  {
    slug: "stat534-hw7-mvn-covariance",
    title: "Multivariate Normal Sampling & Covariance Estimation",
    summary:
      "Empirical covariance; Cholesky factor sampling; BLAS transforms; 10k simulations; sample covariance closely matches target as a correctness check.",
    tags: ["C/C++", "GSL", "Statistical Computing"],
    year: 2025,
    repo: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/hwk7",
  },
  {
    slug: "stat534-final-mpi-volleyball",
    title: "MPI Volleyball Match Simulator",
    summary:
      "13-process simulation (referee + 12 players) with point-to-point messages; rally probabilities and GSL RNG; compact payloads; referee scoring/set logic; clean termination; portable mpic++ Makefile.",
    tags: ["C/C++, MPI", "Parallel Computing"],
    year: 2025,
    repo: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/Stat534Final",
  },

  // Stat 311 — Statistical Computing in R
  {
    slug: "rstat-lab2-nyc-flights-delays",
    title: "NYC Flight Delays — Exploratory Analysis (R Lab 2)",
    summary:
      "Tidyverse EDA on NYC flights; histogram binning, route filters (e.g., LAX), Feb-SFO subset medians/IQR; ranked months by average dep/arr delays to surface seasonality.",
    tags: ["R", "tidyverse", "ggplot2", "dplyr", "Data Wrangling"],
    year: 2025,
    repo: "https://github.com/VarunP3000/RStatisticalComputing/blob/main/RStatisticalComputing/StatsLab2.pdf",
  },
  {
    slug: "rstat-lab3-random-simulations",
    title: "Random Simulations — Coins & Dice (R Lab 3)",
    summary:
      "Sampling with/without replacement; empirical proportions/variance/SD; manual vs built-ins; increased N from 60→1000 shows convergence toward theoretical variance.",
    tags: ["R", "Base R", "Probability and Statistics"],
    year: 2025,
    repo: "https://github.com/VarunP3000/RStatisticalComputing/blob/main/RStatisticalComputing/StatsLab3.pdf",
  },
  {
    slug: "rstat-lab4-hot-hand-fast-food",
    title: "Hot Hand & Fast-Food Normality (R Lab 4)",
    summary:
      "Kobe 2009 Finals streaks vs independent shooter (p=0.45); higher variance in simulation, no ‘hot hand’. Fast-food calories from fat; normal overlays + QQ/qqnormsim checks.",
    tags: ["R", "tidyverse", "ggplot2", "Probability and Statistics"],
    year: 2025,
    repo: "https://github.com/VarunP3000/RStatisticalComputing/blob/main/RStatisticalComputing/StatsLab4.pdf",
  },
  {
    slug: "rstat-lab5-sampling-distributions",
    title: "Sampling Distributions & Population Proportions (R Lab 5)",
    summary:
      "15k resamples via infer + tidyverse; p-hat for beliefs about scientists; compared sampling distributions at n=10/50/100 — more normal, mean→truth, SE↓ as n↑.",
    tags: ["R", "tidyverse", "Probability and Statistics"],
    year: 2025,
    repo: "https://github.com/VarunP3000/RStatisticalComputing/blob/main/RStatisticalComputing/StatsLab5.pdf",
  },
  {
    slug: "rstat-lab6-texting-and-driving",
    title: "Texting & Driving — Hypothesis Testing and Confidence Intervals (R Lab 6)",
    summary:
      "YRBSS texting-while-driving proportion; 99%/95% CIs; margin-of-error vs population p; test H0: p=0.05, p≈0.00014 → strong evidence p>0.05.",
    tags: ["R", "tidyverse", "Probability and Statistics"],
    year: 2025,
    repo: "https://github.com/VarunP3000/RStatisticalComputing/blob/main/RStatisticalComputing/StatsLab6.pdf",
  },
  {
    slug: "rstat-lab7-weight-vs-activity",
    title: "Weight vs Physical Activity — Two-Sample Inference (R Lab 7)",
    summary:
      "YRBSS: active (≥3d/wk) vs inactive; boxplots; conditions checked; two-sample t-tests. p≈0.0002 (two-sided) / 0.0001 (one-sided) → reject H0 at 5%.",
    tags: ["R", "tidyverse", "Probability and Statistics"],
    year: 2025,
    repo: "https://github.com/VarunP3000/RStatisticalComputing/blob/main/RStatisticalComputing/StatsLab7.pdf",
  },
  {
    slug: "rstat-lab8-linear-regression-freedom",
    title: "Linear Regression — Personal Freedom vs Expression Control (R Lab 8)",
    summary:
      "Modeled pf_score ~ pf_expression_control; slope 0.54, intercept 4.28; diagnostics: linearity, normal residuals, constant variance; plots confirm validity.",
    tags: ["R", "tidyverse", "ggplot2", "Probability and Statistics"],
    year: 2025,
    repo: "https://github.com/VarunP3000/RStatisticalComputing/blob/main/RStatisticalComputing/StatsLab8.pdf",
  },
];
