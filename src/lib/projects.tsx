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
    slug: "cse163-us-traffic-pollution-accidents",
    title: "U.S. Traffic, Pollution & Accidents — Data Analysis",
    summary:
      "Merged 3 nationwide datasets (33M congestion rows, 46-col accidents, 22-col pollution) into a unified state-day panel; explored how congestion relates to accidents, seasonality, and emissions using statistical models & geospatial plots.",
    tags: ["Python", "pandas", "scikit-learn", "GeoPandas", "Data Cleaning", "Regression", "Classification"],
    year: 2025,
    repo: "https://github.com/VarunP3000/U.S.-Traffic-Pollution-Accidents-Data-Analysis",
    problem:
      "Understand how congestion, weather, and pollution interact—e.g., what conditions create high accident risk, how congestion varies by month/state, and whether high-traffic areas contribute more to emissions.",
    approach: [
      "Cleaned & merged accidents, pollution, and congestion datasets by grouping to (date, state) and aggregating severity, delays, emissions, and accident counts.",
      "Built EDA: scatterplots, weather-wise accident curves, month-level trends, state-level geospatial maps for congestion and pollutants.",
      "Trained models: linear regression for accidents, logistic regression for congestion buckets, and DecisionTree/RandomForest/GradientBoosting for pollutant prediction.",
    ],
    results: [
      "Accidents peak at medium congestion under clear weather; delay metrics are right-skewed and weakly correlated.",
      "Congestion stable across states except low-population regions (e.g., ND, ME); peaks in late spring/summer.",
      "SO₂ and NO₂ show the strongest (but still weak) pollution–congestion alignment; GradientBoosting achieved lowest RMSE for all pollutants.",
    ],
    limitations:
      "State-level aggregation hides local patterns; weather bucketing simplifies rich categories; correlations can’t confirm causation.",
  },
  {
    slug: "llm-uncertainty-quantification",
    title: "LLM Uncertainty Quantification",
    summary:
      "Tool that runs multiple LLMs on the same dataset and reports confidence, calibration (ECE), and an aggregated ensemble prediction. Includes CSV uploads and a simple React + Node/Python workflow.",
    tags: ["Python", "pandas", "Machine Learning", "Git/GitHub"],
    year: 2025,
    repo: "https://github.com/VarunP3000/ConfidenceScoringProject",
    live: "",
    problem:
      "Single-model LLM outputs can be unstable, and teams often need a clearer read on confidence before trusting model decisions.",
    approach: [
      "Built a UI for uploading CSV prompts and entering a Hugging Face token.",
      "Backend (Node + Python) runs multiple models on inputted dataset, collects their confidence scores, and computes ECE.",
      "Combined the model outputs into a simple confidence-weighted ensemble and exported results as JSON/CSV.",
    ],
    results: [
      "More stable predictions compared to using any one model alone.",
      "ECE highlighted the confidence levels of certain models.",
      "Easy to plug in and remove models without changing code because of the UI flow.",
    ],
    limitations:
      "Slow when many models are selected; model outputs can still be correlated since they’re trained on similar data.",
  },
  {
    slug: "stock-market-cpi-to-sp500-fullstack",
    title: "Stock Market Prediction — CPI → S&P 500",
    summary:
      "Full-stack app that tests how inflation data (CPI) relates to short-term S&P 500 returns. Includes a FastAPI backend for models and a Next.js dashboard for running scenarios.",
    tags: ["Python", "scikit-learn", "pandas", "Machine Learning", "Git/GitHub", "FastAPI", "Next.js"],
    year: 2025,
    repo: "https://github.com/VarunP3000/stock-cost-of-living-app",
    live: "",
    problem:
      "People often assume inflation moves the market, but it’s unclear which CPI components matter or how strong the relationship actually is.",
    approach: [
      "Pulled and merged CPI categories with S&P 500 returns, then created lagged features to test delayed effects.",
      "Trained simple regression models (Ridge, ElasticNet, Gradient Boosting) with proper time-series splits.",
      "Built a dashboard where you can adjust weights or run 'what-if' scenarios and see model outputs instantly.",
    ],
    results: [
      "ElasticNet ended up being the most stable baseline across different time windows.",
      "Some CPI categories showed predictable lag patterns, but only within certain periods.",
      "The dashboard made it easy to see how model predictions changed under different inflation assumptions.",
    ],
    limitations:
      "Market regimes shift a lot, and CPI alone can’t explain most of the movement. Some categories also don’t have enough clean historical data to rely on.",
  },
  {
    slug: "info330-social-media-addiction-db",
    title: "Student Social Media Addiction — Relational DB & Analytics",
    summary:
      "Designed a normalized SQL Server schema and analytic queries to study how student social media use relates to sleep, mental health, relationships, and academics.",
    tags: ["SQL Server", "T-SQL", "Database Design", "ERD", "Data Modeling"],
    year: 2025,
    repo: "https://github.com/<your-username>/SocialMediaRelationships", // TODO: replace with actual repo URL
    problem:
      "Model and query student wellness data (social media usage, sleep, mental health, relationships, demographics) to uncover patterns in digital addiction.",
    approach: [
      "Converted a logical ERD (Student, Sleep, AcademicLevel, Platform, StudentSocialMediaUsage, AddictionAssessment, MentalHealth, Relationships, Country, RelationshipStatus) into a physical SQL Server schema with surrogate PKs, FKs, NOT NULL, DEFAULT, UNIQUE, and CHECK constraints.",
      "Resolved the many-to-many Student↔Platform relationship via an associative entity (StudentSocialMediaUsage) storing avg_daily_hours, and derived interpretable attributes like sleep_performance ('Good'/'Poor').",
      "Chose data types and constraints to match semantics (e.g., INT ages 10–40, DECIMAL/NUMERIC for hours, 1–10 problematic_use_score, BIT addiction_indicator with validation).",
      "Wrote multi-CTE T-SQL queries to compare wellness by academic level, gender, and platform: aggregating sleep, addiction, conflicts_over_social_media, and overall_mental_health_score.",
    ],
    results: [
      "Showed high school students had the lowest sleep (~5.5 hours), highest poor-sleep rate, and highest addiction scores, while undergrad/grad students looked healthier on average.",
      "Found similar average conflict counts and mental health scores across genders in the sample, challenging assumptions about gendered differences in online conflict.",
      "Identified WhatsApp, Instagram, and TikTok as highest-risk platforms, with addiction rates up to 100% in the sample and ~5–6 average daily hours among addicted users.",
    ],
    limitations:
      "Survey is self-reported and snapshot-only (no temporal attributes), ERD vs physical design required some cardinality simplifications, and storing only a primary platform plus a fixed addiction cutoff reduces behavioral nuance.",
  },  
  // --- CSE 373 — DS&A (Java) ---
  {
    slug: "cse373-deques-arraydeque-linkeddeque",
    title: "Deques (ArrayDeque & LinkedDeque)",
    summary:
      "Implemented two representations of the Deque abstract data type: an array-based version with front/back indices and a node-based version using sentinel nodes. Focused on correctness, invariants, and performance.",
    tags: ["Java", "Data Structures", "JUnit/Testing"],
    year: 2024,
    repo: "https://github.com/VarunP3000/DataStructures-Algorithms/tree/main/src/main/java/deques",
    problem:
      "The provided ArrayListDeque implementation was simple but had degraded performance for some operations. I needed to build two alternative representations of Deque and compare their behavior.",
    approach: [
      "Completed ArrayDeque by using an array where elements do not need to start at index 0 and are positioned based on the front and back fields.",
      "Identified and fixed the bug in ArrayDeque by reading test failures, forming hypotheses, and tracing through the resize logic.",
      "Implemented LinkedDeque using two sentinel nodes and maintained the required invariants before and after each method.",
      "Used the provided DequeTests and checkInvariants method to debug edge cases.",
    ],
    results: [
      "ArrayDeque supported constant-time front/back operations without shifting elements, improving performance over ArrayListDeque.",
      "LinkedDeque maintained correct next/prev relationships and used memory proportional to the number of elements.",
      "Passed the full test suite, including confusingTest and the runtime experiments.",
    ],
    limitations:
      "Does not include additional features like fail-fast iterators; focused strictly on the Deque interface methods defined in the project.",
  },  
  {
    slug: "cse373-autocomplete-4-data-structures",
    title: "Autocomplete",
    summary:
      "Built multiple implementations of the autocomplete operation and compared how different data structures handle prefix queries, sorting, and returning the top matches.",
    tags: ["Java", "Data Structures", "Algorithms", "JUnit/Testing"],
    year: 2024,
    repo: "https://github.com/VarunP3000/DataStructures-Algorithms/tree/main/src/main/java/autocomplete",
    problem:
      "Efficiently support the autocomplete operation: given a prefix, return terms that start with that prefix, sorted by their weights.",
    approach: [
      "Started with simpler baseline implementations to establish correctness before focusing on performance.",
      "Implemented additional data structures to support faster lookup of all terms that share a given prefix.",
      "Ensured results were sorted correctly and returned in the proper order required by the specification.",
      "Used the provided tests, including corner cases and larger input files, to verify behavior.",
    ],
    results: [
      "All implementations produced correct matches for prefix queries and returned results in the required sorted order.",
      "Observed clear differences in runtime depending on how each structure stored and accessed its terms.",
    ],
    limitations:
      "Focused only on prefix-based matching and the operations defined in the project; does not support extensions such as approximate matching or full-text search.",
  },  
  {
    slug: "cse373-priority-queues-minpq",
    title: "Priority Queues (MinPQ)",
    summary:
      "Implemented multiple versions of a priority queue and compared how their different representations affect operations like remove-smallest and changing priorities.",
    tags: ["Java", "Data Structures", "Algorithms", "JUnit/Testing"],
    year: 2024,
    repo: "https://github.com/VarunP3000/DataStructures-Algorithms/tree/main/src/main/java/minpq",
    problem:
      "Design a priority queue that efficiently supports removing the smallest item and updating priorities while maintaining correct ordering.",
    approach: [
      "Began with a simple array-based implementation to establish correct behavior for core priority queue operations.",
      "Implemented a binary heap using the array representation described in class, maintaining the heap-order property through swimming and sinking.",
      "Used tests to confirm that the heap structure preserved the correct parent/child relationships after each operation.",
      "Compared runtimes of the different implementations to see how representation impacts efficiency.",
    ],
    results: [
      "The heap-based implementation handled remove-smallest and priority updates much more efficiently than a simple array scan.",
      "Maintained the heap-order property across a variety of operations and input sizes.",
    ],
    limitations:
      "Focused only on the binary heap representation and the operations covered in the course; does not include additional priority queue variants.",
  },  
  {
    slug: "cse373-shortest-paths-and-seam-carving",
    title: "Shortest Paths & Seam Carving",
    summary:
      "Implemented shortest-path algorithms on weighted directed graphs and applied the same ideas to find minimum-energy seams for image resizing.",
    tags: ["Java", "Graphs", "Dynamic Programming", "Shortest Paths"],
    year: 2024,
    repo: "https://github.com/VarunP3000/DataStructures-Algorithms/tree/main/src/main/java/graphs",
    problem:
      "Given a weighted directed graph, compute the shortest path from a source to all reachable vertices. Then use those techniques to find a minimum-energy vertical seam in an image.",
    approach: [
      "Represented graphs using the adjacency-list structure provided in the project.",
      "Implemented Dijkstra’s algorithm by repeatedly relaxing edges until all shortest paths were finalized.",
      "Implemented the shortest-path algorithm for directed acyclic graphs using a topological ordering and a single pass of edge relaxations.",
      "Applied dynamic programming to the seam-carving task by treating each pixel as a vertex with an energy value and computing the minimum-energy path from top to bottom.",
      "Compared runtimes of the different shortest-path approaches as graphs and images grew in size.",
    ],
    results: [
      "Both graph-based algorithms produced correct shortest paths under the project’s test cases.",
      "The dynamic-programming approach for seam carving was significantly faster because the underlying graph is always a directed acyclic graph.",
    ],
    limitations:
      "Focused only on single-seam removal and the algorithms required in the specification; did not extend to removing multiple seams or modifying the energy function.",
  },  
  // --- CSE/STAT 416 — AI/ML (Python) ---
  {
    slug: "c416-house-prices",
    title: "House Prices Prediction",
    summary:
      "Used pandas and linear regression to explore housing data and build models that predict house prices.",
    tags: ["Python", "scikit-learn", "pandas"],
    year: 2025,
    repo: "https://github.com/VarunP3000/PythonAI-MLProjects/blob/main/hw0.ipynb",
    problem:
      "Given a dataset of houses, the goal was to use the input columns to predict the price column without leaking information from the validation or test sets.",
    approach: [
      "Explored the dataset using pandas to understand the rows, columns, and basic statistics.",
      "Split the data into train, validation, and test sets using the provided code.",
      "Trained two linear regression models in scikit-learn: one using a small set of basic features and one using a larger set of advanced features.",
      "Evaluated both models by computing the RMSE on the training and validation sets.",
    ],
    results: [
      "The model with the advanced features performed better on the validation data, so it was used to compute the final test error.",
    ],
    limitations:
      "Only linear regression was used; no other feature sets or modeling choices were explored beyond the assignment requirements.",
  },
  {
    slug: "c416-sentiment-analysis-amazon",
    title: "Sentiment Analysis (Amazon Reviews)",
    summary:
      "Used product review data from Amazon.com, turned reviews into word-count features, and trained logistic regression models to predict whether a review is positive or negative.",
    tags: ["Python", "pandas", "scikit-learn", "Logistic Regression"],
    year: 2025,
    repo: "https://github.com/VarunP3000/PythonAI-MLProjects/blob/main/SentimentAnalysisWithLogisticRegression.ipynb",
    problem:
      "Given food product reviews from Amazon, predict whether the sentiment of each review is positive (+1) or negative (-1) using the review text.",
    approach: [
      "Loaded the food_products.csv data into a pandas DataFrame and created a sentiment column from the rating values (+1 or -1).",
      "Removed punctuation from the review text and used scikit-learn's CountVectorizer to build word-count features.",
      "Split the data into training, validation, and test sets with train_test_split.",
      "Trained a majority class classifier as a baseline and then fit a logistic regression sentiment model in scikit-learn.",
      "Computed validation accuracy with accuracy_score, built a confusion matrix, and inspected the most positive and most negative words using the model coefficients.",
      "Trained additional logistic regression models with different L2 regularization strengths and recorded their train and validation accuracies in a table."
    ],
    results: [
      "The logistic regression sentiment model achieved higher validation accuracy than the majority class classifier.",
      "Words such as 'great' and 'best' had large positive coefficients, while words such as 'not' and 'bland' had large negative coefficients, matching how we expect people to write positive and negative reviews."
    ],
    limitations:
      "Uses only word-count features and logistic regression; does not include more advanced text processing or models beyond what was required in the assignment."
  },  
  {
    slug: "c416-loan-default-trees-rf",
    title: "Loan Safety with Decision Trees and a Small Random Forest",
    summary:
      "Used LendingClub loan data to predict whether a loan is safe (+1) or risky (-1) with decision trees and a simple random forest, and compared training vs validation accuracy across different tree depths.",
    tags: ["Python", "pandas", "scikit-learn", "Decision Trees"],
    year: 2025,
    repo: "https://github.com/VarunP3000/PythonAI-MLProjects/blob/main/LoanSafety.ipynb",
    problem:
      "Given LendingClub data with features like grade, home ownership, purpose, term, and debt-to-income ratio, predict whether a loan is a safe loan (+1) or a risky loan (-1).",
    approach: [
      "Loaded lending-club-data.csv, created the safe_loans label from bad_loans, and explored features such as grade and home_ownership.",
      "Selected the assignment’s feature list and used pd.get_dummies to one-hot encode categorical columns for sklearn.",
      "Trained DecisionTreeClassifier models with different max_depth values, and used GridSearchCV over max_depth and min_samples_leaf to tune early-stopping settings.",
      "Implemented a small RandomForest416 class that fits multiple trees on bootstrap samples and predicts by majority vote, then compared its train/validation accuracy to a single tree."
    ],
    results: [
      "Deeper trees fit the training data very well but did not always improve validation accuracy, showing overfitting at large depths.",
      "The RandomForest416 model generally gave better validation accuracy than a single decision tree at similar depths.",
      "Features based on grade, sub_grade, home_ownership, purpose, and term became usable once expanded into one-hot encoded columns."
    ],
    limitations:
      "Used only decision trees and a small random forest on train/validation splits (no separate held-out test set or additional model families beyond the assignment scope)."
  },
  {
    slug: "c416-cifar10-nets-a-b-c-d",
    title: "CIFAR-10 Image Classification (NetA–NetD)",
    summary:
      "Built and trained several PyTorch neural networks (NetA–NetD) on CIFAR-10 to classify 32×32 color images into 10 classes, comparing fully connected and convolutional models using GPU training.",
    tags: ["Python", "PyTorch", "Deep Learning", "Neural Networks"],
    year: 2025,
    repo: "https://github.com/VarunP3000/PythonAI-MLProjects/blob/main/DeepLearning.ipynb",
    problem:
      "Use PyTorch to design neural networks that reach the required validation accuracy on CIFAR-10 and understand how different architectures affect image classification performance.",
    approach: [
      "Loaded the CIFAR-10 train/validation sets with torchvision, applied the given tensor/normalize transforms, and used DataLoader batches on GPU.",
      "Implemented NetA and NetB as fully connected networks that flatten each image and pass it through one or two hidden layers with ReLU.",
      "Implemented NetC as a small convolutional network with a conv → ReLU → max-pool block, then a fully connected layer to 10 outputs.",
      "Designed NetD with two convolution layers and two fully connected layers, following the assignment rules, and trained all nets with cross-entropy loss, Adam, and the provided train/accuracy/plot_history helpers."
    ],
    results: [
      "NetA and NetB reached validation accuracies in the low-50% range, showing what simple fully connected networks can do on CIFAR-10.",
      "NetC improved validation accuracy to around mid-60%, and NetD reached about 71% best validation accuracy.",
      "Training vs validation curves showed higher training accuracy than validation, illustrating overfitting but still clear gains from adding convolution layers."
    ],
    limitations:
      "Did not deeply tune hyperparameters or explore additional regularization beyond what was required."
  },
  {
    slug: "c416-kmeans-from-scratch-wikipedia-tfidf",
    title: "K-Means from Scratch (Wikipedia, TF-IDF)",
    summary:
      "Implemented k-means in NumPy and applied it to TF-IDF vectors of ~5.9k Wikipedia biographies to study clustering behavior under different inits and K values.",
    tags: ["Python", "NumPy", "Clustering", "Machine Learning"],
    year: 2025,
    repo: "https://github.com/VarunP3000/PythonAI-MLProjects/blob/main/KMeansWithTextData.ipynb",
    problem:
      "Cluster unlabeled Wikipedia biography articles and analyze how initialization, number of clusters K, and multiple runs affect cluster quality and stability.",
    approach: [
      "Vectorized the core k-means steps (assign_clusters, revise_centroids, heterogeneity, main loop) over sparse TF-IDF features with L2 normalization.",
      "Compared vanilla random initialization vs k-means++ and ran multiple seeds per setting, using heterogeneity to quantify local minima.",
      "Swept over different K values (e.g., 2, 10, 25), plotting K vs heterogeneity and using an elbow-style heuristic to judge tradeoffs.",
      "Interpreted clusters by inspecting nearest documents and top TF-IDF terms per centroid to reveal topics (politicians, athletes, artists, academics, etc.)."
    ],
    results: [
      "k-means++ plus multi-seed restarts consistently achieved lower heterogeneity than naive random init, indicating better local optima.",
      "Larger K values broke broad clusters (e.g., generic ‘researchers’ or ‘politicians’) into more coherent subgroups like footballers, golfers, orchestra conductors, and visual artists.",
      "Cluster inspections showed reasonably pure topic groups for moderate K, with nearest-neighbor bios and top words matching intuitive themes."
    ],
    limitations:
      "Uses standard k-means, so assumes roughly spherical clusters in Euclidean space and requires K to be chosen; results are also tied to TF-IDF bag-of-words (no semantic embeddings)."
  },
  {
    slug: "c416-twitter-topic-modeling-nmf",
    title: "Twitter Topic Modeling (NMF)",
    summary:
      "Modeled ~119k April 30, 2020 COVID-era tweets with TF-IDF + NMF to discover latent topics, inspect top words per topic, and analyze tweet–topic weights and outliers.",
    tags: ["Python", "NMF/Topic Modeling", "Machine Learning"],
    year: 2025,
    repo: "https://github.com/VarunP3000/PythonAI-MLProjects/blob/main/RecommendationWithText.ipynb",
    problem:
      "Discover unsupervised themes in short COVID-related tweets and understand how strongly each tweet loads onto those topics, including detection of unusual outlier groups.",
    approach: [
      "Used scikit-learn TF-IDF (max_df=0.95) over pre-cleaned tweets (English only, lowercased, no URLs/punct/stopwords/common COVID tokens).",
      "Fit NMF with k=5 (init='nndsvd') to get tweet-topic loadings and word-topic weights; wrote helpers to rank top words per topic.",
      "Assigned each tweet to its dominant topic via argmax on the projection matrix, then counted topic popularity across the corpus.",
      "Refit NMF with k=3 for 3D visualization of tweets in topic space and flagged outliers with high Topic-2 weights, inspecting their raw text via .unique()."
    ],
    results: [
      "5-topic model produced interpretable themes (e.g., case/death counts, health/support, US/China/politics, apps to slow spread), with one topic clearly dominating tweet assignments.",
      "3-topic model merged related themes but preserved a similar structure and revealed a tight outlier cluster dominated by app/self-reporting/symptom-tracking style tweets.",
      "Outlier analysis showed those tweets share rare but highly weighted terms, explaining why NMF isolates them as a distinct, high-Topic-2 group."
    ],
    limitations:
      "Bag-of-words TF-IDF over a single day of tweets; topics and outliers are sensitive to k, initialization, and short, sparse tweet text, so themes may drift across reruns or different time windows."
  }
  ,
  // --- INFO 201 — Data Science & Informatics Foundations (R) ---
  {
    slug: "info201-capstone-stock-retirement",
    title: "INFO 201 Capstone — Stock Market as an Economic Indicator",
    summary:
      "RMarkdown analysis tying S&P 500 trends to CPI, tax burden, and housing costs (US vs non-US), with fully reproducible data cleaning, merging, and visualization.",
    tags: ["R", "tidyverse", "Data Wrangling", "Visualization"],
    year: 2025,
    repo: "https://github.com/mojipao/Stock-Market-Retirement",
    problem:
      "Test whether the S&P 500 can be used as a proxy for real-world economic conditions by relating it to inflation (CPI), tax costs, and housing costs in the US and abroad.",
    approach: [
      "Ingested three public datasets in R (Yahoo Finance S&P 500, FRED CPIAUCSL, Kaggle cost-of-living data) and converted all to a yearly panel with consistent Year keys.",
      "Used tidyverse pipelines to clean and aggregate data: yearly averages for CPI and S&P 500, grouped cost-of-living metrics by Year × Country, and derived numeric cost components (housing, healthcare, education, transport, and tax) from percentage fields.",
      "Tagged rows as US vs Non-US, merged CPI and S&P 500 only where appropriate (S&P 500 kept for US only), and enforced one row per country-year via summarize(across(..., mean, na.rm = TRUE)).",
      "Built ggplot2 visuals for four research questions: S&P 500 vs CPI (scatter + lm line), S&P 500 vs tax cost (dual bar chart + trend), US vs non-US housing cost trends (line plot), and US housing vs S&P 500 (scaled overlay)."
    ],
    results: [
      "Found a positive correlation between S&P 500 closing values and average CPI, consistent with the idea that strong markets often co-occur with periods of rising prices and economic expansion.",
      "Observed that S&P 500 levels trend upward over time while tax costs (in dollar terms) fluctuate without a clear long-run pattern, suggesting tax burden is shaped more by policy and macro conditions than by market performance alone.",
      "Showed that US housing costs are noticeably more volatile than non-US housing costs from 2000–2023, with sharper booms and dips likely tied to US-specific cycles and policy shocks.",
      "For the US, housing costs and the S&P 500 both rise over the long run and sometimes move together (especially post-2012), but the relationship is non-linear and clearly influenced by distinct drivers (rates, demand, supply, etc.)."
    ],
    limitations:
      "Correlation-only yearly aggregates (no causal identification); S&P 500 used as a single broad index with no sector breakdown; cross-country housing comparisons ignore purchasing-power differences; some cost-of-living and tax fields contain missing data that were averaged but not imputed, and external macro factors (rates, employment, policy regimes) were left out of the models."
  },
  // --- STAT/BIOSTAT 534 — Statistical Computing (UW) ---
  {
    slug: "stat534-hw3-mc3-model-selection",
    title: "Stochastic Model Selection via MC3",
    summary:
      "MC3 over add/remove-one neighborhoods; rcdd linearity checks; MH with neighbor-count correction; compared to greedy selections.",
    tags: ["R", "Statistical Computing", "Bayesian Methods"],
    year: 2025,
    repo: "https://github.com/VarunP3000/R-C-Stats-Projects/tree/main/Stat534Projects/Stat534HW3",
    problem: "Explore model space without exhaustive enumeration.",
    approach: [
      "MC3 proposals; acceptance with neighborhood correction; chains diagnostics; rcdd validity filters.",
    ],
    results: [
      "Stochastic search recovered high-scoring subsets similar to greedy but with alternatives for correlated features.",
    ],
    limitations: "Chain mixing depends on proposal tuning.",
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
    approach: [
      "Mode finding; Laplace evidence; MH initialized at mode; log-accept tracking; parallel runs.",
    ],
    results: [
      "Laplace close to MH posterior means; acceptance stable with tuned proposals.",
    ],
    limitations: "Univariate only; Gaussian priors.",
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
    approach: [
      "Linear algebra kernels via BLAS/LAPACK & GSL; attention to row/col major; tests vs R.",
    ],
    results: [
      "Numeric parity with R; strong runtime characteristics on medium-n.",
    ],
    limitations: "CPU-only; no batching across many models.",
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
    approach: [
      "MPI ranks for roles; GSL RNG; serve/rally probabilities; referee tracks scoring and sets; termination messages.",
    ],
    results: [
      "Deterministic logs and correct match rules; portable build with mpic++.",
    ],
    limitations: "No network failures; simplified rally model.",
  },
];
