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
      "End-to-end pipeline to score LLM outputs with confidence thresholds and calibration. CSV→prompt ingestion, chained calls, and confidence scoring with tunable thresholds. Compared raw vs calibrated confidence using ECE/accuracy and simple prompt/model ensembling. Interfaces split across Node.js orchestration and Python scoring with reproducible metrics.",
    skills: ["Python", "Node.js", "TypeScript", "pandas", "Metrics"],
  },
  {
    id: "stock-fullstack-cpi-spx",
    title: "Stock Market Prediction — Full‑Stack (CPI → S&P 500)",
    year: 2025,
    topics: ["AI/ML", "Full-Stack"],
    tags: ["Python", "scikit-learn", "pandas", "Machine Learning", "Git/GitHub"],
    github: "https://github.com/VarunP3000/stock-cost-of-living-app", // TODO replace with repo link
    description:
      "Full-stack forecasting app linking CPI features to S&P 500 returns. FastAPI backend serves regression/ensemble endpoints with backtesting; Next.js frontend provides scenario exploration with adjustable weights. Focus on clear API contracts, CORS-safe deployment, and readable plots/metrics.",
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
      "Built a circular-array deque with wraparound front/back indices and amortized resizing; fixed a tricky resize/wrap bug uncovered by failing tests. Implemented a sentinel‑based doubly‑linked deque (two sentinels) guaranteeing O(1) add/remove at both ends. Benchmarked against an ArrayList‑backed deque and explained O(1) deque removes vs O(n) ArrayList.remove(0).",
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
      "Implemented SequentialSearchAutocomplete (linear scan) and BinarySearchAutocomplete (sorted list + Collections.binarySearch + rightward sweep). Built TernarySearchTreeAutocomplete using TST nodes (left/mid/right) with end‑of‑term markers and DFS collect. Compared to a TreeSet baseline (NavigableSet.ceiling/tailSet) with targeted tests and worst‑ vs average‑case analysis.",
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
      "Wrote UnsortedArrayMinPQ with PriorityNode pairs (linear contains/changePriority; min scan for peekMin/removeMin). Implemented HeapMinPQ via java.util.PriorityQueue and a remove‑then‑reinsert changePriority. Designed an optimized heap + index map achieving near O(log n) changePriority and fast membership checks; contrasted with DoubleMapMinPQ reference.",
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
      "Built two seam‑graph representations: AdjacencyListSeamFinder (materialized Pixel[][] with edges) and GenerativeSeamFinder (neighbors on demand). Implemented ToposortDAGSolver (DFS postorder → relax in topo order) and used Dijkstra; also worked with Bellman‑Ford, SPFA, and A* APIs. Added a pure DP seam finder and compared runtime/allocations across five approaches.",
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
      "Prepared tabular data (imputation/encoding/splits); established linear baselines and regularized models. Tuned hyperparameters on validation and monitored RMSE on train/val/test to check generalization and avoid leakage. Interpreted coefficients and summarized failure modes.",
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
      "Engineered polynomial/sqrt features and standardized the design matrix. Performed alpha (lambda) sweeps with validation curves; contrasted Ridge shrinkage vs LASSO sparsity on held‑out RMSE. Visualized coefficient paths and non‑zero patterns.",
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
      "Cleaned reviews, built bag‑of‑words features with CountVectorizer, and split into train/val/test. Trained logistic regression (baseline = majority class), reported accuracy/confusion matrix, and produced class probabilities. Explored L2 regularization and identified most positive/negative terms.",
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
      "One‑hot encoded categorical fields; addressed class imbalance; tuned depth/min‑samples via GridSearchCV. Implemented bootstrap aggregation for a custom Random Forest. Plotted learning curves and importances; compared high‑variance trees vs lower‑variance forest.",
    skills: ["Python", "scikit-learn"],
  },
  {
    id: "c416-cifar10",
    title: "CIFAR‑10 Image Classification (Net A/B/C/D)",
    year: 2025,
    topics: ["AI/ML", "Deep Learning"],
    tags: ["Python", "PyTorch", "Deep Learning", "Neural Networks"],
    github: "https://github.com/VarunP3000/PythonAI-MLProjects/blob/main/DeepLearning.ipynb",
    description:
      "Implemented MLP and CNN variants (conv → ReLU → pooling → dropout) with efficient data loaders and augmentations. Trained on GPU, tracked accuracy/loss per epoch, applied early stopping and LR scheduling, and examined misclassifications to guide tweaks.",
    skills: ["PyTorch", "Python"],
  },
  {
    id: "c416-kmeans-tfidf",
    title: "K‑Means from Scratch (Wikipedia, TF‑IDF)",
    year: 2025,
    topics: ["AI/ML", "Unsupervised"],
    tags: ["Python", "NumPy", "Clustering", "Machine Learning"],
    github: "https://github.com/VarunP3000/PythonAI-MLProjects/blob/main/KMeansWithTextData.ipynb",
    description:
      "Implemented assignment/recompute loop with inertia/heterogeneity metrics and convergence checks. Added k‑means++ initialization and multiple restarts; evaluated K via elbow/heterogeneity trends. Clustered TF‑IDF features and interpreted clusters via top‑weight terms.",
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
      "Vectorized cleaned tweets with TF‑IDF (stopwords, lowercasing, punctuation removal). Factorized with NMF (k=5 then k=3), extracted top words per topic, mapped tweet→topic weights, and identified an outlier cluster along Topic 2; discussed themes.",
    skills: ["Python", "scikit-learn"],
  },

  // INFO 201 — Data Science & Informatics Foundations (R)
  {
    id: "info201-ps01-basicr",
    title: "PS01 — Basic R (variables, logic, strings, loops, functions)",
    year: 2024,
    topics: ["Data Science", "R"],
    tags: ["R", "tidyverse"],
    github: "https://github.com/VarunP3000/RDataScienceProjects/blob/main/Info201Assignments/INFO201ps01.R",
    description:
      "Computed derived quantities (seconds/year, age in seconds, relativistic mass), practiced boolean logic, manipulated text with stringr, and wrote parameterized functions with defaults (e.g., greeting, RemoveDigits). Implemented classic loops and validated results.",
    skills: ["R", "stringr"],
  },
  {
    id: "info201-ps02-vectors",
    title: "PS02 — Vectors, Vectorization & Lists",
    year: 2024,
    topics: ["Data Science", "R"],
    tags: ["R"],
    github: "https://github.com/VarunP3000/RDataScienceProjects/blob/main/Info201Assignments/INFO201ps02.R",
    description:
      "Built and sliced numeric/character vectors; applied vectorized transforms and logical indexing. Designed a Student Support calculator via vectorized conditions and named vectors; used lists for structured data and wrote a dice simulator with a test hook.",
    skills: ["R"],
  },
  {
    id: "info201-ps03-rmd",
    title: "PS03 — R Markdown, Filesystems & Control Flow",
    year: 2024,
    topics: ["Data Science", "R"],
    tags: ["R"],
    github: "https://github.com/VarunP3000/RDataScienceProjects/blob/main/Info201Assignments/INFO201ps03.Rmd",
    description:
      "Authored an RMarkdown report (lists, code blocks, images); explained working directories and rendered reproducible output. Enumerated files with list.files/file.info and generated type‑aware sentences; classified files by extension with concise ifelse pipelines.",
    skills: ["R", "RMarkdown"],
  },
  {
    id: "info201-ps04-lifeexp-flights",
    title: "PS04 — Life Expectancy & NYC Flights",
    year: 2024,
    topics: ["Data Science", "R"],
    tags: ["R", "tidyverse"],
    github: "https://github.com/VarunP3000/RDataScienceProjects/blob/main/Info201Assignments/INFO201ps04.Rmd",
    description:
      "Analyzed life‑expectancy data (NA tallies, per‑country growth, regional summaries) and built labeled scatterplots. Cleaned NYC flights, analyzed delays to SEA, worst destinations, monthly trends, and derived mph speeds with sanity checks.",
    skills: ["R", "tidyverse", "ggplot2"],
  },
  {
    id: "info201-ps05-co2-gapminder",
    title: "PS05 — CO2, Gapminder & Fertility vs Life Expectancy",
    year: 2024,
    topics: ["Data Science", "R"],
    tags: ["R", "tidyverse"],
    github: "https://github.com/VarunP3000/RDataScienceProjects/blob/main/Info201Assignments/INFO201ps05.html",
    description:
      "Audited country identifiers (ISO‑2/3, names); contrasted total vs per‑capita emissions across USA/CHN/IND/JPN/BRA; computed regional means (1960 vs 2016). Modeled orange tree growth using lag; traced country paths with ggplot2 and geom_path.",
    skills: ["R", "tidyverse", "ggplot2"],
  },
  {
    id: "info201-ps06-co2-temp",
    title: "PS06 — CO2 vs Temperature (Scripps, HadCRUT, UAH)",
    year: 2024,
    topics: ["Data Science", "R"],
    tags: ["R", "tidyverse"],
    github: "https://github.com/VarunP3000/RDataScienceProjects/blob/main/Info201Assignments/INFO201ps06.Rmd",
    description:
      "Cleaned Scripps monthly CO2 and created a continuous time axis; computed anomaly vs a pre‑industrial baseline from HadCRUT; merged yearly CO2 with temperature anomalies; compared UAH vs surface temps with decade‑colored trends.",
    skills: ["R", "tidyverse", "ggplot2"],
  },
  {
    id: "info201-infolab5",
    title: "InfoLab5 — Data Frames, Subsetting & Simple Analytics",
    year: 2024,
    topics: ["Data Science", "R"],
    tags: ["R"],
    github: "https://github.com/VarunP3000/RDataScienceProjects/blob/main/Info201Assignments/InfoLab5.Rmd",
    description:
      "Built a Seahawks results data frame with engineered features; loaded HR salaries to compute raises and identify max raise; demonstrated vectorized arithmetic, logical filters, and tidy printing for quick EDA.",
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
      "Structured a reproducible R project with data ingestion, tidy transforms, and clearly labeled plots in RMarkdown. Implemented time‑series style computations and yearly aggregations for retirement planning; documented assumptions and repo organization.",
    skills: ["R", "tidyverse", "RMarkdown"],
  },

  // STAT/BIOSTAT 534 — Statistical Computing (UW)
  {
    id: "s534-hw1-bayes-lm",
    title: "HW1 — Bayesian Linear Models: Log‑Determinant & Marginal Likelihood",
    year: 2025,
    topics: ["Statistical Computing"],
    tags: ["R", "Statistical Computing"],
    github: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/Stat534HW1",
    description:
      "Implemented eigendecomposition‑based logdet and closed‑form log marginal likelihood pipelines for normal linear regression with conjugate priors. Verified on estrogen‑receptor data and reproduced textbook examples.",
    skills: ["R", "Matrix Algebra"],
  },
  {
    id: "s534-hw2-greedy-logistic",
    title: "HW2 — Greedy Variable Selection for Logistic Regression",
    year: 2025,
    topics: ["Statistical Computing"],
    tags: ["R", "Statistical Computing"],
    github: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/Stat534HW2",
    description:
      "Wrapped glm(binomial) with convergence/NA guards; computed AIC/BIC; implemented forward/backward greedy searches that add/remove one predictor minimizing AIC. Applied to a 60‑feature dataset; aligned forward AIC and BIC selections.",
    skills: ["R"],
  },
  {
    id: "s534-hw3-mc3",
    title: "HW3 — Stochastic Model Selection via MC3",
    year: 2025,
    topics: ["Statistical Computing"],
    tags: ["R", "Statistical Computing", "Bayesian Methods"],
    github: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/Stat534HW3",
    description:
      "Implemented MC3 subset selection with add/remove‑one neighbors; filtered valid models via rcdd linearity checks to avoid separation. Used neighbor‑count‑corrected MH and ran multiple chains to compare best subsets/AICs vs greedy search.",
    skills: ["R", "MCMC"],
  },
  {
    id: "s534-hw4-laplace-mh",
    title: "HW4 — Bayesian Univariate Logistic Regression (Laplace + MH)",
    year: 2025,
    topics: ["Statistical Computing"],
    tags: ["R", "Statistical Computing", "Bayesian Methods", "Parallel Computing"],
    github: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/hwk4",
    description:
      "Found posterior modes via Newton–Raphson under N(0,1) priors; computed Laplace log‑evidence and built an MH sampler seeded at the mode. Parallelized 60 univariate fits; summarized posterior means and compared with MLE checks.",
    skills: ["R", "snow", "MASS"],
  },
  {
    id: "s534-hw5-marglik-cpp",
    title: "HW5 — Marginal Likelihood for Linear Regression (C/C++: LAPACK & GSL)",
    year: 2025,
    topics: ["Statistical Computing", "Systems"],
    tags: ["C/C++", "LAPACK", "GSL", "Statistical Computing", "Makefiles", "Performance Optimization"],
    github: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/hwk5",
    description:
      "Two high‑performance implementations of log marginal likelihood for [1|A]: LAPACKE (dposv solve; log‑det via eigens) and GSL (LU with lndet). Engineered GEMM/identity‑add/solve/log‑det with careful memory ownership; matched R baseline.",
    skills: ["C/C++", "LAPACK/LAPACKE", "GSL", "Make"],
  },
  {
    id: "s534-hw6-rec-det-topk",
    title: "HW6 — Recursive Determinant & Top‑K Regression Search",
    year: 2025,
    topics: ["Statistical Computing"],
    tags: ["C/C++", "LAPACK", "Algorithms", "Statistical Computing"],
    github: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/hwk6",
    description:
      "Coded a pure recursive determinant via Laplace expansion with disciplined memory cleanup; evaluated on 10×10 banded matrices. Extended AddRegression to maintain a descending‑likelihood singly linked list with de‑duplication and tail trims for top‑K.",
    skills: ["C/C++", "Algorithms"],
  },
  {
    id: "s534-hw7-mvn-cov",
    title: "HW7 — Multivariate Normal Sampling & Covariance Estimation",
    year: 2025,
    topics: ["Statistical Computing"],
    tags: ["C/C++", "GSL", "Statistical Computing"],
    github: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/hwk7",
    description:
      "Computed empirical covariance Σ, built an MVN sampler using Cholesky (Σ = ΨΨᵀ), drew Z~N(0,I), and formed X=ΨZ via BLAS. Simulated 10k draws and compared sample covariance to Σ as a correctness check; shipped portable Makefiles and utilities.",
    skills: ["C/C++", "GSL", "BLAS"],
  },
  {
    id: "s534-final-mpi-volleyball",
    title: "Final — MPI Volleyball Match Simulator",
    year: 2025,
    topics: ["Systems", "Parallel"],
    tags: ["C/C++, MPI", "Parallel Computing"],
    github: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/Stat534Final",
    description:
      "13‑process simulation: referee (rank 0) and 12 players; ball passed via point‑to‑point messages. Encoded serve/rally probabilities with a three‑hit cap and uniform teammate/opponent selection via GSL RNG. Referee handled scoring and set logic; logged events; portable mpic++ Makefile.",
    skills: ["C/C++", "MPI", "Make"],
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
              I’m an undergraduate at the University of Washington studying Informatics (ML specialization) and ACMS (Data Science & Statistics). I enjoy building practical ML systems, clear APIs, and interfaces that explain model behavior. My work spans data engineering, time-series forecasting, uncertainty/confidence scoring, and full-stack development.
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
