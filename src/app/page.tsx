"use client";

import React, { useMemo, useState } from "react";
import { JSX } from "react/jsx-dev-runtime";

// Types
export type Project = {
  id: string;
  title: string;
  year: number;
  topics: string[];
  tags: string[];
  github?: string;
  description?: string;
  skills?: string[];
};

// Data
const PROJECTS: Project[] = [
  // Personal / Extra-curricular
  {
    id: "llm-uq",
    title: "LLM Uncertainty Quantification (Confidence & Calibration)",
    year: 2025,
    topics: ["AI/ML", "Data Science"],
    tags: ["Python", "pandas", "Machine Learning", "Git/GitHub"],
    github: "https://github.com/VarunP3000/ConfidenceScoringProject", // TODO replace with repo link
    description:
      "Built an end-to-end pipeline that scores LLM outputs with calibrated confidence. I ingest CSVs into prompts, chain model calls, and compute confidence scores with tunable thresholds. I compare raw versus calibrated predictions using ECE and accuracy, and I support simple prompt/model ensembling. The system separates orchestration in Node.js from scoring in Python so results are reproducible and easy to extend.",
    skills: ["Python", "Node.js", "TypeScript", "pandas", "Metrics"],
  },
  {
    id: "stock-fullstack-cpi-spx",
    title: "Stock Market Prediction — Full-Stack (CPI → S&P 500)",
    year: 2025,
    topics: ["AI/ML", "Full-Stack"],
    tags: ["Python", "scikit-learn", "pandas", "Machine Learning", "Git/GitHub"],
    github: "https://github.com/VarunP3000/stock-cost-of-living-app", // TODO replace with repo link
    description:
      "Designed a full-stack forecasting app that links CPI features to subsequent S&P 500 returns. The FastAPI backend serves regression and ensemble endpoints with backtesting and quantile outputs. A Next.js front end lets users run scenarios and adjust model weights. I emphasized clear API contracts, CORS-safe deployment, and readable metrics and plots.",
    skills: ["Python", "FastAPI", "scikit-learn", "pandas", "React", "Next.js", "TypeScript"],
  },

  // CSE 373 — Data Structures & Algorithms (Java)
  {
    id: "cse373-deques",
    title: "Deques (ArrayDeque & LinkedDeque)",
    year: 2024,
    topics: ["DS&A"],
    tags: ["Java", "Data Structures", "JUnit/Testing", "Performance Optimization"],
    github: "https://github.com/VarunP3000/DataStructures-Algorithms/tree/main/src/main/java/deques",
    description:
      "Implemented a circular-array deque with wraparound indices and amortized resizing, then fixed a tricky wrap/resize bug surfaced by failing tests. I also built a sentinel-based doubly linked deque that guarantees O(1) adds and removes at both ends. Finally, I benchmarked against an ArrayList-backed deque and explained why deque removes are O(1) while ArrayList.remove(0) is O(n), writing edge-case tests to lock in the behavior.",
    skills: ["Java", "JUnit", "Data Structures"],
  },
  {
    id: "cse373-autocomplete-4ds",
    title: "Autocomplete (4 Data Structures)",
    year: 2024,
    topics: ["DS&A"],
    tags: ["Java", "Data Structures", "Algorithms", "JUnit/Testing"],
    github: "https://github.com/VarunP3000/DataStructures-Algorithms/tree/main/src/main/java/autocomplete",
    description:
      "Built two baseline autocompletes: a sequential scan and a binary-search-based approach over a sorted list. Then I implemented a Ternary Search Tree (TST) autocomplete with end-of-term markers and a DFS collector for results. I compared all variants against a TreeSet baseline, added unit tests for simple and adversarial cases, and analyzed worst-case versus average-case lookup costs.",
    skills: ["Java", "Algorithms", "Testing"],
  },
  {
    id: "cse373-minpq",
    title: "Priority Queues (MinPQ)",
    year: 2024,
    topics: ["DS&A"],
    tags: ["Java", "Data Structures", "Algorithms", "Performance Optimization", "JUnit/Testing"],
    github: "https://github.com/VarunP3000/DataStructures-Algorithms/tree/main/src/main/java/minpq",
    description:
      "Implemented UnsortedArrayMinPQ using PriorityNode pairs with linear membership and change-priority and a min scan for removal. I then built HeapMinPQ using java.util.PriorityQueue and handled changePriority by remove-then-reinsert. Finally, I designed an optimized binary heap with an index map to achieve near O(log n) changePriority and fast membership, and contrasted it with a DoubleMapMinPQ reference under benchmarks.",
    skills: ["Java", "Priority Queues", "Performance"],
  },
  {
    id: "cse373-shortestpaths-seam",
    title: "Shortest Paths & Seam Carving",
    year: 2024,
    topics: ["DS&A", "Algorithms"],
    tags: ["Java", "Graphs", "Dynamic Programming", "Shortest Paths", "Algorithms"],
    github: "https://github.com/VarunP3000/DataStructures-Algorithms/tree/main/src/main/java/graphs",
    description:
      "Constructed two seam-graph representations: a materialized adjacency list of pixels and a generative graph that yields neighbors on demand. I implemented a DAG topological solver (DFS postorder followed by relax) and used Dijkstra; I also worked with Bellman-Ford, SPFA, and A*. I added a pure dynamic-programming seam finder and compared runtime and allocations across all five approaches.",
    skills: ["Java", "Graphs", "DP"],
  },

  // CSE/STAT 416 — AI/ML (Python)
  {
    id: "c416-house-prices",
    title: "House Prices Prediction",
    year: 2025,
    topics: ["AI/ML", "Data Science"],
    tags: ["Python", "scikit-learn", "pandas", "Machine Learning"],
    github: "https://github.com/VarunP3000/PythonAI-MLProjects/blob/main/hw0.ipynb",
    description:
      "Prepared a tabular dataset with careful imputation, encoding, and train/validation/test splits. I established linear baselines and regularized models and tuned hyperparameters using validation curves. I monitored RMSE to check generalization and potential leakage, and I interpreted coefficients to explain feature effects and failure modes.",
    skills: ["Python", "scikit-learn", "pandas"],
  },
  {
    id: "c416-ridge-lasso",
    title: "Ridge & LASSO Regression",
    year: 2025,
    topics: ["AI/ML"],
    tags: ["Python", "scikit-learn", "Machine Learning"],
    github: "https://github.com/VarunP3000/PythonAI-MLProjects/blob/main/Ridge%26Lasso.ipynb",
    description:
      "Engineered polynomial and square-root features and standardized the design matrix inside reproducible pipelines. I swept α (λ) values and plotted validation curves to select regularization strength. I contrasted Ridge shrinkage against LASSO sparsity on held-out RMSE and visualized coefficient paths to discuss bias–variance trade-offs.",
    skills: ["Python", "scikit-learn"],
  },
  {
    id: "c416-sentiment-amazon",
    title: "Sentiment Analysis (Amazon Reviews)",
    year: 2025,
    topics: ["AI/ML"],
    tags: ["Python", "scikit-learn", "Machine Learning"],
    github: "https://github.com/VarunP3000/PythonAI-MLProjects/blob/main/SentimentAnalysisWithLogisticRegression.ipynb",
    description:
      "Cleaned Amazon review text and built bag-of-words features using CountVectorizer with an appropriate train/validation/test split. I trained logistic regression (with the majority-class baseline for context) and reported accuracy, a confusion matrix, and class probabilities. I explored L2 regularization, traced important coefficients, and highlighted the most positive and negative terms.",
    skills: ["Python", "scikit-learn"],
  },
  {
    id: "c416-loan-default",
    title: "Loan Default Risk (Trees & Custom Random Forest)",
    year: 2025,
    topics: ["AI/ML"],
    tags: ["Python", "scikit-learn", "Machine Learning"],
    github: "https://github.com/VarunP3000/PythonAI-MLProjects/blob/main/LoanSafety.ipynb",
    description:
      "One-hot encoded categorical fields and addressed class imbalance while creating robust splits. I tuned decision-tree depth and minimum sample parameters via GridSearchCV and implemented a custom Random Forest with bootstrap aggregation. I plotted learning curves and feature importances and compared a high-variance tree to a lower-variance forest.",
    skills: ["Python", "scikit-learn"],
  },
  {
    id: "c416-cifar10",
    title: "CIFAR-10 Image Classification (Net A/B/C/D)",
    year: 2025,
    topics: ["AI/ML", "Deep Learning"],
    tags: ["Python", "PyTorch", "Deep Learning", "Neural Networks"],
    github: "https://github.com/VarunP3000/PythonAI-MLProjects/blob/main/DeepLearning.ipynb",
    description:
      "Implemented several architectures ranging from MLPs to deeper CNN stacks using PyTorch with efficient data loaders and augmentations. I trained on GPU, tracked accuracy and loss per epoch, and used early stopping and learning-rate scheduling. I reached the target validation accuracy and examined misclassifications to guide further tweaks.",
    skills: ["PyTorch", "Python"],
  },
  {
    id: "c416-kmeans-tfidf",
    title: "K-Means from Scratch (Wikipedia, TF-IDF)",
    year: 2025,
    topics: ["AI/ML", "Unsupervised"],
    tags: ["Python", "NumPy", "Clustering", "Machine Learning"],
    github: "https://github.com/VarunP3000/PythonAI-MLProjects/blob/main/KMeansWithTextData.ipynb",
    description:
      "Implemented k-means from scratch with the standard assign-and-recompute loop, inertia metrics, and convergence checks. I added k-means++ initialization and multiple random restarts to improve stability. I evaluated K via elbow and heterogeneity trends, clustered TF-IDF features, and interpreted clusters by top-weight terms and examples.",
    skills: ["Python", "NumPy"],
  },
  {
    id: "c416-twitter-nmf",
    title: "Twitter Topic Modeling (NMF)",
    year: 2025,
    topics: ["AI/ML", "Unsupervised"],
    tags: ["Python", "NMR/Topic Modeling", "Machine Learning"],
    github: "https://github.com/VarunP3000/PythonAI-MLProjects/blob/main/RecommendationWithText.ipynb",
    description:
      "Vectorized cleaned tweets with TF-IDF, including stopword handling, case normalization, and punctuation removal. I factorized the matrix using NMF with k=5 and k=3, extracted top words per topic, and mapped tweet-to-topic weights. I identified the dominant topic per tweet, surfaced an outlier cluster along one topic dimension, and summarized the themes.",
    skills: ["Python", "scikit-learn"],
  },

  // INFO 201 — Data Science & Informatics Foundations (R)
  {
    id: "info201-ps01-basicr",
    title: "Basic R (variables, logic, strings, loops, functions)",
    year: 2024,
    topics: ["Data Science", "R"],
    tags: ["R", "tidyverse"],
    github: "https://github.com/VarunP3000/RDataScienceProjects/blob/main/Info201Assignments/INFO201ps01.R",
    description:
      "Wrote small R programs that compute derived quantities and practice boolean logic, string manipulation, and basic functions. I used stringr for templating, replacement, casing, and character counts. I implemented classic loops such as running sums and factorials, formatted outputs, and validated results with simple checks.",
    skills: ["R", "stringr"],
  },
  {
    id: "info201-ps02-vectors",
    title: "Vectors, Vectorization & Lists",
    year: 2024,
    topics: ["Data Science", "R"],
    tags: ["R"],
    github: "https://github.com/VarunP3000/RDataScienceProjects/blob/main/Info201Assignments/INFO201ps02.R",
    description:
      "Practiced vector creation, slicing, and fully vectorized transforms using base R. I designed a student-support calculator with named vectors and logical indexing to allocate awards. I also used lists for structured data and wrote a dice simulator with a test hook to verify behavior.",
    skills: ["R"],
  },
  {
    id: "info201-ps03-rmd",
    title: "R Markdown, Filesystems & Control Flow",
    year: 2024,
    topics: ["Data Science", "R"],
    tags: ["R"],
    github: "https://github.com/VarunP3000/RDataScienceProjects/blob/main/Info201Assignments/INFO201ps03.Rmd",
    description:
      "Authored a reproducible R Markdown report that mixes narrative with code and images. I explained working directories and relative paths and demonstrated file enumeration with list.files and file.info. I classified files by extension using concise ifelse pipelines and showcased inline code within the report.",
    skills: ["R", "RMarkdown"],
  },
  {
    id: "info201-ps04-lifeexp-flights",
    title: "Life Expectancy & NYC Flights",
    year: 2024,
    topics: ["Data Science", "R"],
    tags: ["R", "tidyverse"],
    github: "https://github.com/VarunP3000/RDataScienceProjects/blob/main/Info201Assignments/INFO201ps04.Rmd",
    description:
      "Analyzed life-expectancy data by tallying missing values, computing per-country growth, and summarizing by region. I built labeled scatterplots and demonstrated the equivalence of mean-of-differences and difference-of-means using tidy verbs. I also cleaned NYC flights, analyzed delays to SEA, surfaced monthly trends, and derived mph speeds with sanity checks.",
    skills: ["R", "tidyverse", "ggplot2"],
  },
  {
    id: "info201-ps05-co2-gapminder",
    title: "CO2, Gapminder & Fertility vs Life Expectancy",
    year: 2024,
    topics: ["Data Science", "R"],
    tags: ["R", "tidyverse"],
    github: "https://github.com/VarunP3000/RDataScienceProjects/blob/main/Info201Assignments/INFO201ps05.html",
    description:
      "Audited country identifiers across ISO-2, ISO-3, and names and diagnosed Namibia’s missing ISO-2 due to the “NA” sentinel. I contrasted total versus per-capita CO₂ emissions across major countries and computed regional means for 1960 and 2016 with tidyverse tools. I modeled orange tree growth using lag, plotted continent-level life-expectancy means, and traced country paths with ggplot2.",
    skills: ["R", "tidyverse", "ggplot2"],
  },
  {
    id: "info201-ps06-co2-temp",
    title: "CO2 vs Temperature (Scripps, HadCRUT, UAH)",
    year: 2024,
    topics: ["Data Science", "R"],
    tags: ["R", "tidyverse"],
    github: "https://github.com/VarunP3000/RDataScienceProjects/blob/main/Info201Assignments/INFO201ps06.Rmd",
    description:
      "Cleaned Scripps monthly CO₂ data, removed implausible rows, and created a continuous time axis to visualize the long-term rise. I derived a pre-industrial baseline from HadCRUT and plotted temperature anomalies relative to that baseline. I aggregated CO₂ to yearly means, merged with surface and satellite temperature anomalies, and compared decade-colored trend lines.",
    skills: ["R", "tidyverse", "ggplot2"],
  },
  {
    id: "info201-infolab5",
    title: "Data Frames, Subsetting & Simple Analytics",
    year: 2024,
    topics: ["Data Science", "R"],
    tags: ["R"],
    github: "https://github.com/VarunP3000/RDataScienceProjects/blob/main/Info201Assignments/InfoLab5.Rmd",
    description:
      "Built a tidy data frame of Seahawks results, engineered a margin and win indicator, and performed targeted subsetting. I loaded HR salaries and computed raises, counted recipients, and identified the maximum raise and the corresponding employee. I demonstrated vectorized arithmetic, logical filters, and tidy printing for quick exploratory analysis.",
    skills: ["R"],
  },
  {
    id: "info201-capstone-retirement",
    title: "Capstone — Stock Market Retirement (Repo)",
    year: 2024,
    topics: ["Data Science", "R"],
    tags: ["R", "tidyverse"],
    github: "https://github.com/mojipao/Stock-Market-Retirement",
    description:
      "Structured a reproducible R project for retirement-style stock analysis using tidy ingestion, transformations, and clearly labeled plots. I assembled the work in an R Markdown report with transparent assumptions. I implemented yearly aggregations and time-series style computations and organized the repository for easy review and extension.",
    skills: ["R", "tidyverse", "RMarkdown"],
  },

  // STAT/BIOSTAT 534 — Statistical Computing (UW)
  {
    id: "s534-hw1-bayes-lm",
    title: "Bayesian Linear Models: Log-Determinant & Marginal Likelihood",
    year: 2025,
    topics: ["Statistical Computing"],
    tags: ["R", "Statistical Computing"],
    github: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/Stat534HW1",
    description:
      "Implemented eigendecomposition-based log-determinant and a closed-form marginal-likelihood pipeline for conjugate normal linear models. I built the full computation for arbitrary predictor subsets on centered and standardized designs. I verified results on the estrogen-receptor dataset and reproduced textbook examples with matching numeric values.",
    skills: ["R", "Matrix Algebra"],
  },
  {
    id: "s534-hw2-greedy-logistic",
    title: "Greedy Variable Selection for Logistic Regression",
    year: 2025,
    topics: ["Statistical Computing"],
    tags: ["R", "Statistical Computing"],
    github: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/Stat534HW2",
    description:
      "Wrote AIC/BIC helpers and robust glm wrappers that handle convergence and missing data. I implemented forward and backward greedy searches that add or remove one predictor to minimize AIC, including a safe-skip mechanism for non-convergent fits. On a 60-feature dataset, the forward AIC reached a compact model and closely aligned with forward BIC selections.",
    skills: ["R"],
  },
  {
    id: "s534-hw3-mc3",
    title: "Stochastic Model Selection via MC3",
    year: 2025,
    topics: ["Statistical Computing"],
    tags: ["R", "Statistical Computing", "Bayesian Methods"],
    github: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/Stat534HW3",
    description:
      "Implemented MC3 subset selection over add/remove-one neighborhoods and filtered valid models using rcdd linearity checks to avoid separation. I used a Metropolis–Hastings scheme with neighbor-count correction and ran multiple chains. I compared the best subsets and AICs against the greedy approach from HW2 for stability.",
    skills: ["R", "MCMC"],
  },
  {
    id: "s534-hw4-laplace-mh",
    title: "Bayesian Univariate Logistic Regression (Laplace + MH)",
    year: 2025,
    topics: ["Statistical Computing"],
    tags: ["R", "Statistical Computing", "Bayesian Methods", "Parallel Computing"],
    github: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/hwk4",
    description:
      "Found the posterior mode for univariate logistic regression under N(0,1) priors using Newton–Raphson. I computed a Laplace approximation to the evidence and built an MH sampler initialized at the mode, recording acceptance on the log scale. I parallelized 60 fits with snow and summarized posterior means with optional MLE sanity checks.",
    skills: ["R", "snow", "MASS"],
  },
  {
    id: "s534-hw5-marglik-cpp",
    title: "Marginal Likelihood for Linear Regression (C/C++: LAPACK & GSL)",
    year: 2025,
    topics: ["Statistical Computing", "Systems"],
    tags: ["C/C++", "LAPACK", "GSL", "Statistical Computing", "Makefiles", "Performance Optimization"],
    github: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/hwk5",
    description:
      "Implemented two high-performance versions of the linear-model marginal likelihood in C/C++: one using LAPACKE and one using GSL. I engineered GEMM, identity-add, solve, and log-det computations with careful memory ownership and row/column-major handling. The implementations matched an R baseline and passed the spec check for a known subset.",
    skills: ["C/C++", "LAPACK/LAPACKE", "GSL", "Make"],
  },
  {
    id: "s534-hw6-rec-det-topk",
    title: "Recursive Determinant & Top-K Regression Search",
    year: 2025,
    topics: ["Statistical Computing"],
    tags: ["C/C++", "LAPACK", "Algorithms", "Statistical Computing"],
    github: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/hwk6",
    description:
      "Coded a pure recursive determinant using Laplace expansion with exact base cases and disciplined memory cleanup. I extended a data structure to maintain a descending-likelihood singly linked list with de-duplication and a bounded size for top-K models. I enumerated all ≤2-predictor models on a benchmark dataset and wrote the top results to disk.",
    skills: ["C/C++", "Algorithms"],
  },
  {
    id: "s534-hw7-mvn-cov",
    title: "Multivariate Normal Sampling & Covariance Estimation",
    year: 2025,
    topics: ["Statistical Computing"],
    tags: ["C/C++", "GSL", "Statistical Computing"],
    github: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/hwk7",
    description:
      "Computed the empirical covariance matrix from a real dataset and used a Cholesky factorization to sample from a multivariate normal. I generated standard-normal draws, transformed them via BLAS operations, and reconstructed samples with the correct covariance. I simulated 10,000 draws, recomputed the sample covariance, and compared it elementwise to the target as a correctness check.",
    skills: ["C/C++", "GSL", "BLAS"],
  },
  {
    id: "s534-final-mpi-volleyball",
    title: "MPI Volleyball Match Simulator",
    year: 2025,
    topics: ["Systems", "Parallel"],
    tags: ["C/C++, MPI", "Parallel Computing"],
    github: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/Stat534Final",
    description:
      "Built a 13-process volleyball simulation with a referee and twelve players that pass a “ball” by point-to-point messages. I encoded serve and rally probabilities with a three-hit cap, used a GSL RNG for uniform selection, and defined compact message payloads and event reports. The referee implemented scoring and set logic (to 25, win-by-2; best-of-five), logged actions, and terminated players cleanly with a portable mpic++ Makefile.",
    skills: ["C/C++", "MPI", "Make"],
  },
  //Stat 311 Statistical Computing in R
  {
    id: "rstat-statslab2-nycflights",
    title: "NYC Flight Delays — Exploratory Analysis (R Lab 2)",
    year: 2025,
    topics: ["Statistical Computing", "R"],
    tags: ["R", "tidyverse", "ggplot2", "dplyr", "Data Wrangling"],
    github: "https://github.com/VarunP3000/RStatisticalComputing/blob/main/RStatisticalComputing/StatsLab2.pdf",
    description:
      "Explored departure/arrival delays in the NYC flights dataset using tidyverse. Compared histograms with different bin counts, filtered and summarized routes (e.g., LAX-only delays), and created a February SFO subset to analyze medians/IQR by origin. Wrapped up by ranking months by average departure and arrival delays to surface seasonal patterns.",
    skills: ["R", "tidyverse", "ggplot2"]
  },
  {
    id: "rstat-statslab3-random-simulations",
    title: "Random Simulations — Coins & Dice (R Lab 3)",
    year: 2025,
    topics: ["Statistical Computing", "R"],
    tags: ["R", "Simulation", "Random Sampling", "Probability", "Base R"],
    github: "https://github.com/VarunP3000/RStatisticalComputing/blob/main/RStatisticalComputing/StatsLab3.pdf",
    description:
      "Simulated coin tosses and dice rolls in R to study sampling with and without replacement. Computed empirical proportions, sample variance, and standard deviation manually and via built-in R functions to validate formulas. Increased simulation size from 60 to 1000 rolls to observe convergence toward theoretical variance and reduced variability.",
    skills: ["R", "Simulation", "Probability"]
  },
  {
    id: "rstat-statslab4-kobe-fastfood",
    title: "Hot Hand & Fast-Food Normality (R Lab 4)",
    year: 2025,
    topics: ["Statistical Computing", "R"],
    tags: ["R", "tidyverse", "ggplot2", "Probability", "Simulation", "EDA", "QQ Plot"],
    github: "https://github.com/VarunP3000/RStatisticalComputing/blob/main/RStatisticalComputing/StatsLab4.pdf",
    description:
      "Analyzed Kobe Bryant’s 2009 Finals shot streaks with calc_streak and compared against an independent-shooter simulation (p=0.45), finding higher variance and mean streak length in the simulation and no ‘hot hand’. Then explored fast-food nutrition: histograms of calories from fat for McDonald’s vs. Dairy Queen, overlaid a normal curve using sample mean/SD, and used QQ plots/qqnormsim to assess normality.",
    skills: ["R", "tidyverse", "ggplot2"]
  },
  {
    id: "rstat-statslab5-sampling-distributions",
    title: "Sampling Distributions & Population Proportions (R Lab 5)",
    year: 2025,
    topics: ["Statistical Computing", "R"],
    tags: ["R", "infer", "tidyverse", "Simulation", "Sampling", "Central Limit Theorem"],
    github: "https://github.com/VarunP3000/RStatisticalComputing/blob/main/RStatisticalComputing/StatsLab5.pdf",
    description:
      "Simulated repeated random samples to study the sampling distribution of proportions using the infer and tidyverse packages. Estimated p-hat for beliefs about scientists’ work benefiting society and visualized 15,000 samples with histograms. Compared sampling distributions at n = 10, 50, and 100, showing that as sample size increases, the shape becomes more normal, the mean approaches the true population proportion (0.2), and the standard error decreases.",
    skills: ["R", "tidyverse", "infer", "Simulation"]
  },
  {
    id: "rstat-statslab6-texting-driving",
    title: "Texting & Driving — Hypothesis Testing and Confidence Intervals (R Lab 6)",
    year: 2025,
    topics: ["Statistical Computing", "R"],
    tags: ["R", "infer", "Hypothesis Testing", "Confidence Intervals", "Proportions", "tidyverse"],
    github: "https://github.com/VarunP3000/RStatisticalComputing/blob/main/RStatisticalComputing/StatsLab6.pdf",
    description:
      "Analyzed Youth Risk Behavior Survey (YRBSS) data to estimate the proportion of high schoolers who text while driving. Computed 99% and 95% confidence intervals for the true proportion using the infer package and visualized the margin of error as a function of population proportion. Conducted hypothesis tests (H₀: p=0.05) and found a p-value ≈ 0.00014, providing strong evidence that more than 5% of high schoolers text while driving.",
    skills: ["R", "infer", "tidyverse", "Hypothesis Testing"]
  },
  {
    id: "rstat-statslab7-weight-activity",
    title: "Weight vs Physical Activity — Two-Sample Inference (R Lab 7)",
    year: 2025,
    topics: ["Statistical Computing", "R"],
    tags: ["R", "infer", "Hypothesis Testing", "t-test", "Data Visualization", "tidyverse"],
    github: "https://github.com/VarunP3000/RStatisticalComputing/blob/main/RStatisticalComputing/StatsLab7.pdf",
    description:
      "Used the Youth Risk Behavior Survey (YRBSS) to test whether physically active high schoolers (≥3 days/week) weigh more than inactive peers. Created boxplots comparing groups, verified inference conditions, and ran two-sample t-tests using infer. Obtained a p-value ≈ 0.0002 for a two-sided test and 0.0001 for a one-sided alternative, leading to rejection of the null hypothesis at the 5% significance level.",
    skills: ["R", "infer", "tidyverse", "Statistical Inference"]
  },
  {
    id: "rstat-statslab8-linear-regression",
    title: "Linear Regression — Personal Freedom vs Expression Control (R Lab 8)",
    year: 2025,
    topics: ["Statistical Computing", "R"],
    tags: ["R", "Linear Regression", "tidyverse", "ggplot2", "Model Diagnostics"],
    github: "https://github.com/VarunP3000/RStatisticalComputing/blob/main/RStatisticalComputing/StatsLab8.pdf",
    description:
      "Performed simple linear regression on the Human Freedom Index dataset to model personal freedom (pf_score) as a function of expression control (pf_expression_control). Found a strong positive slope (0.54) and intercept (4.28), indicating higher freedom scores with greater expression control. Verified regression assumptions—linearity, normal residuals, and constant variance—through fitted vs. residual plots and histograms, confirming model validity.",
    skills: ["R", "tidyverse", "ggplot2", "Regression Analysis"]
  },              
];

function classNames(...xs: Array<string | false | null | undefined>): string {
  return xs.filter(Boolean).join(" ");
}

interface SectionProps {
  id: string;
  title: string;
  children: React.ReactNode;
}

function Section({ id, title, children }: SectionProps) {
  return (
    <section id={id} className="scroll-mt-24 py-12 sm:py-16">
      <div className="mx-auto max-w-6xl px-4">
        <h2 className="text-2xl sm:text-3xl font-semibold tracking-tight mb-6">{title}</h2>
        {children}
      </div>
    </section>
  );
}

interface ProjectCardProps {
  p: Project;
}

function ProjectCard({ p }: ProjectCardProps) {
  return (
    <div className="rounded-2xl border border-zinc-200/60 dark:border-zinc-800 bg-white dark:bg-zinc-900 shadow-sm p-5">
      <div className="flex items-start justify-between gap-4">
        <h3 className="text-lg font-semibold leading-snug">{p.title}</h3>
        <span className="text-xs rounded-full border border-zinc-200 dark:border-zinc-800 px-2 py-1">{p.year}</span>
      </div>

      {/* Topics row */}
      <ul className="mt-3 flex flex-wrap gap-2">
        {p.topics?.map((topic: string) => (
          <li key={topic} className="text-xs rounded-full bg-zinc-100 dark:bg-zinc-800 px-2 py-1">
            {topic}
          </li>
        ))}
      </ul>

      {/* GitHub link */}
      {p.github ? (
        <div className="mt-4">
          <a href={p.github} target="_blank" className="text-sm underline underline-offset-4 hover:no-underline">GitHub Repository</a>
        </div>
      ) : null}

      {/* Paragraph description */}
      {p.description ? (
        <p className="mt-4 text-sm text-zinc-700 dark:text-zinc-300 leading-relaxed">{p.description}</p>
      ) : null}

      {/* Skills/Software */}
      {p.skills?.length ? (
        <div className="mt-4">
          <div className="text-xs uppercase tracking-wide text-zinc-500 mb-1">Skills & Software</div>
          <ul className="flex flex-wrap gap-2">
            {p.skills.map((s: string) => (
              <li key={s} className="text-xs rounded-full border border-zinc-200 dark:border-zinc-800 px-2 py-1">
                {s}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {/* Secondary tags */}
      {p.tags?.length ? (
        <div className="mt-4">
          <div className="text-xs uppercase tracking-wide text-zinc-500 mb-1">Concepts & Tech</div>
          <ul className="flex flex-wrap gap-2">
            {p.tags.map((tag: string) => (
              <li key={tag} className="text-xs rounded-full bg-zinc-100 dark:bg-zinc-800 px-2 py-1">
                {tag}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

export default function PortfolioSite(): JSX.Element {
  const [query, setQuery] = useState<string>("");
  const [activeTag, setActiveTag] = useState<string>("All");

  const tags = useMemo<string[]>(() => {
    const t = new Set<string>(["All"]);
    PROJECTS.forEach((p) => p.tags.forEach((x: string) => t.add(x)));
    return Array.from(t);
  }, []);

  const filtered = useMemo<Project[]>(() => {
    const q = query.trim().toLowerCase();
    return PROJECTS.filter((p) => {
      const matchesTag = activeTag === "All" || p.tags.includes(activeTag);
      const haystack = [p.title, p.description ?? "", ...p.tags, ...p.topics]
        .join(" ")
        .toLowerCase();
      const matchesQuery = !q || haystack.includes(q);
      return matchesTag && matchesQuery;
    }).sort((a, b) => b.year - a.year);
  }, [query, activeTag]);

  return (
    <div className="min-h-screen bg-zinc-50 text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100">
      {/* Nav */}
      <header className="sticky top-0 z-40 backdrop-blur border-b border-zinc-200/70 dark:border-zinc-800/70 bg-white/70 dark:bg-zinc-950/50">
        <div className="mx-auto max-w-6xl px-4 h-16 flex items-center justify-between">
          <a href="#about" className="font-semibold tracking-tight">Varun Panuganti</a>
          <nav className="hidden sm:flex items-center gap-6 text-sm">
            <a href="#about" className="hover:opacity-80">About</a>
            <a href="#education" className="hover:opacity-80">Education</a>
            <a href="#experience" className="hover:opacity-80">Experience</a>
            <a href="#projects" className="hover:opacity-80">Projects</a>
            <a href="#contact" className="hover:opacity-80">Contact</a>
            <a href="/Varun_Panuganti_OG_Resume.pdf" className="inline-flex items-center rounded-xl border px-3 py-1.5 text-sm hover:bg-zinc-50 dark:hover:bg-zinc-900">Resume</a>
          </nav>
        </div>
      </header>

      {/* About (moved to top with photo) */}
      <Section id="about" title="About">
        <div className="grid md:grid-cols-3 gap-8 items-center">
          <div className="md:col-span-1">
            <img src="/headshot.jpg" alt="Varun Panuganti" className="w-full rounded-3xl border border-zinc-200 dark:border-zinc-800 object-cover" />
          </div>
          <div className="md:col-span-2 prose prose-zinc dark:prose-invert max-w-none">
            <p>
            I’m Varun Panuganti, an undergraduate at the University of Washington studying Informatics and ACMS (Data Science & Statistics). I am passionate about studying how data and mathematics can be used to make decisions, and using my software engineering skills to efficiently build products that reflect this. I’m currently seeking internships where I can apply my extensive background in Math/Statistics, AI/ML and Data Science, and Software Engineering to build meaningful products.
            </p>
            <p>
              Languages & tools I use most: Python, C/C++, Java, R, SQL, JavaScript/TypeScript, React/Next.js; PyTorch, scikit-learn, pandas, NumPy; BLAS/LAPACK; Flask/FastAPI; MPI; GitHub and testing frameworks.
            </p>
            <div className="mt-4 flex flex-wrap gap-3">
              <a href="/Varun_Panuganti_OG_Resume.pdf" className="rounded-2xl px-4 py-2 border hover:bg-zinc-50 dark:hover:bg-zinc-900">Download Resume</a>
              <a href="https://github.com/VarunP3000" target="_blank" className="rounded-2xl px-4 py-2 border hover:bg-zinc-50 dark:hover:bg-zinc-900">GitHub</a>
              <a href="https://linkedin.com/in/varun-panuganti" target="_blank" className="rounded-2xl px-4 py-2 border hover:bg-zinc-50 dark:hover:bg-zinc-900">LinkedIn</a>
            </div>
          </div>
        </div>
      </Section>

      {/* Education */}
      <Section id="education" title="Education">
        <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-5">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <div className="flex items-center gap-3">
              <img
                src="/UW_logo.png"
                alt="University of Washington logo"
                className="h-10 w-10 object-contain rounded"
              />
              <div>
                <div className="text-base font-semibold">University of Washington (Seattle Campus)</div>
                <div className="text-sm text-zinc-600 dark:text-zinc-400">
                  Double Degree: Informatics (BS) &amp; ACMS (Data Science and Statistics) (BS)
                </div>
              </div>
            </div>
            <div className="text-sm">GPA: <span className="font-medium">3.77</span></div>
          </div>
          <div className="mt-3">
            <a href="/transcript.pdf" className="text-sm underline underline-offset-4">View Transcript</a>
          </div>
        </div>
      </Section>

      {/* Experience */}
      <Section id="experience" title="Work Experience">
        <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-5">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <div className="flex items-center gap-3">
              <img
                src="/iCode_Logo.jpg"
                alt="ICODE logo"
                className="h-10 w-10 object-contain rounded"
              />
              <div>
                <div className="text-base font-semibold">ICODE</div>
                <div className="text-sm text-zinc-600 dark:text-zinc-400">Sammamish, WA</div>
              </div>
            </div>
            <div className="text-sm">July 2024 – April 2025</div>
          </div>
          <ul className="mt-3 list-disc pl-5 space-y-1 text-sm">
            <li>Guided students in designing self-driving robots using VEX and sensors, culminating in a robot that consistently navigated a maze.</li>
            <li>Mentored K–12 students in game development with Python (Pygame) and Unreal Engine, helping them build a 90%-complete racing game with AI opponents and physics simulation.</li>
            <li>Instructed Java fundamentals (OOP, recursion, algorithms, and data structures), leading students to develop a Spring-based application for data processing and visualization.</li>
          </ul>
          <div className="mt-3 text-sm">
            Supervisor: <span className="font-medium">Rhett Davis</span>{" "}
          </div>
        </div>
      </Section>

      {/* Projects */}
      <Section id="projects" title="Selected Projects">
        <div className="flex flex-col sm:flex-row gap-3 sm:items-center sm:justify-between mb-5">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search projects, tags, tech…"
            className="w-full sm:w-80 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-zinc-300 dark:focus:ring-zinc-700"
          />
          <div className="flex flex-wrap gap-2">
            {tags.map((tag: string) => (
              <button
                key={tag}
                onClick={() => setActiveTag(tag)}
                className={classNames(
                  "rounded-xl px-3 py-1.5 text-sm border",
                  activeTag === tag
                    ? "bg-zinc-900 text-white dark:bg-white dark:text-zinc-900 border-zinc-900 dark:border-white"
                    : "border-zinc-200 dark:border-zinc-800 hover:bg-zinc-50 dark:hover:bg-zinc-900"
                )}
              >
                {tag}
              </button>
            ))}
          </div>
        </div>
        <div className="flex flex-col gap-5">
          {filtered.map((p) => (
            <ProjectCard key={p.id} p={p} />
          ))}
        </div>
      </Section>

      {/* Contact */}
      <Section id="contact" title="Contact">
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
          <a href="mailto:varunp5@uw.edu" className="underline underline-offset-4">varunp5@uw.edu</a>
          <a href="https://github.com/VarunP3000" target="_blank" className="underline underline-offset-4">GitHub</a>
          <a href="https://linkedin.com/in/varun-panuganti" target="_blank" className="underline underline-offset-4">LinkedIn</a>
        </div>
      </Section>

      <footer className="py-10 border-t border-zinc-200 dark:border-zinc-800">
        <div className="mx-auto max-w-6xl px-4 text-sm text-zinc-500">
          © {new Date().getFullYear()} Varun Panuganti. Built with React + Tailwind.
        </div>
      </footer>
    </div>
  );
}
