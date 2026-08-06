const experiences = [
  {
    id: "reu",
    years: "MAY — AUG 2026",
    role: "Research Internship - National Science Foundation funded REU",
    org: "Marshall University · Mentored by Dr. Haroon Malik",
    place: "Huntington, West Virginia",
    kicker: "Cross-domain emotion analytics for NLP",
    tools: ["Python", "PyTorch", "scikit-learn", "H200 GPUs"],
    facts: [
      "Co-authoring an IEEE-format manuscript benchmarking five emotion-annotated text datasets, including GoEmotions, SemEval, and MELD.",
      "Labeled 15M+ RateMyProfessor reviews with a six-model ensemble and a 3-of-6 voting rule, sharded across three H200 GPUs in under one hour.",
      "Ran 2,400 grouped cross-validation fits across 490K texts; logistic regression beat RoBERTa on both multi-label datasets while training 111–185× faster.",
    ],
    links: [
      { label: "Research repository", href: "https://github.com/VarunP3000/NFS-REU-Data-Analytics-2026" },
      { label: "2026 REU program", href: "https://www.marshall.edu/reu/2026-research-projects/" },
    ],
    metric: { big: "15M+", label: "reviews labeled" },
    accent: "lime",
  },
  {
    id: "ta",
    years: "MAR — JUN 2026",
    role: "Teaching Assistant",
    org: "University of Washington · INFO 330",
    place: "Seattle, Washington",
    kicker: "Databases & data modeling",
    tools: ["PostgreSQL", "ER diagrams", "Normalization"],
    facts: [
      "Led office hours and quiz sections, helping students reason through and debug complex PostgreSQL queries.",
      "Guided students from ER diagrams to normalized relational schemas with sound keys and constraints.",
      "Collaborated with course staff to evaluate assignments and deliver structured, actionable feedback.",
    ],
    links: [],
    metric: { big: "INFO 330", label: "Databases & data modeling" },
    accent: "violet",
    placeholderLabel: "University of Washington campus photo",
  },
  {
    id: "bot",
    years: "NOV 2025 — MAY 2026",
    role: "Researcher - Data Science",
    org: "Independent research project",
    place: "Engagement modeling & AI content classification",
    kicker: "Finding signal in unstructured conversation",
    tools: ["Pandas", "NumPy", "scikit-learn"],
    facts: [
      "Built a processing pipeline that converts unstructured comment data into analysis-ready features.",
      "Implemented a decision-tree classifier with optimized feature selection and an F1 score of 0.99.",
      "Designed reusable workflows for large-scale sentiment and engagement analysis across datasets.",
    ],
    links: [{ label: "ViralSongEngagementStudy", href: "https://github.com/VarunP3000/ViralSongEngagementStudy" }],
    metric: { big: "0.99", label: "classifier F1" },
    accent: "orange",
  },
  {
    id: "llm",
    years: "OCT 2024 — OCT 2025",
    role: "Researcher - AI/ML",
    org: "University of Washington",
    place: "LLM uncertainty quantification",
    kicker: "Making model confidence legible",
    tools: ["Python", "Node.js", "React"],
    facts: [
      "Engineered a multi-stage LLM annotation pipeline handling 1K+ API calls per day.",
      "Iterated structured FOR / AGAINST / NEUTRAL prompts to improve stance consistency and enable confidence scoring.",
      "Built a full-stack CSV ingestion application supporting 10K+ tokens per run.",
    ],
    links: [{ label: "Confidence scoring project", href: "https://github.com/VarunP3000/ConfidenceScoringProject" }],
    metric: { big: "1K+", label: "API calls / day" },
    accent: "blue",
  },
  {
    id: "icode",
    years: "JUL 2024 — APR 2025",
    role: "Computer Science Instructor",
    org: "iCode",
    place: "Sammamish, Washington",
    kicker: "Teaching students how to build—and debug",
    tools: ["Java", "Python", "VEX Robotics", "Unreal Engine"],
    facts: [
      "Taught Java and object-oriented design through a Spring-based data application.",
      "Instructed project-based courses across Python, VEX robotics, and Unreal Engine.",
      "Made runtime errors less intimidating by teaching systematic debugging habits.",
    ],
    links: [{ label: "Visit iCode Sammamish", href: "https://icodeschool.com/sammamish/" }],
    metric: { big: "4", label: "technical platforms" },
    accent: "pink",
    placeholderLabel: "iCode Sammamish photo",
  },
];

function Arrow() {
  return <span aria-hidden="true">↗</span>;
}

export default function Home() {
  return (
    <main>
      <nav className="nav" aria-label="Primary navigation">
        <a className="wordmark" href="#top">VARUN<span>.</span></a>
        <div className="nav-links">
          <a href="#experience">Experience</a>
          <a href="#contact">Contact</a>
        </div>
      </nav>

      <header className="hero" id="top">
        <div className="hero-copy">
          <p className="eyebrow">RESEARCHER · ENGINEER · EDUCATOR</p>
          <h1>Hi, I’m<br /><em>Varun.</em></h1>
          <p className="intro">I study machine learning and data science at the University of Washington, with a focus on NLP, model evaluation, and useful research tools.</p>
          <div className="hero-actions">
            <a className="primary-link" href="#experience">View my experience</a>
            <a className="text-link" href="mailto:hello@example.com">Get in touch <Arrow /></a>
          </div>
        </div>
        <div className="hero-photo" role="img" aria-label="Placeholder for a portrait of Varun">
          <span>PORTRAIT</span>
          <div className="portrait-mark">VP</div>
          <p>Replace with your photo</p>
        </div>
      </header>

      <section className="experience" id="experience">
        <div className="section-heading">
          <p className="eyebrow">SELECTED EXPERIENCE · 2024—2026</p>
          <h2>Experience</h2>
        </div>

        {experiences.map((item) => (
          <article className={`case ${item.accent}`} id={item.id} key={item.id}>
            <div className="case-index">
              <p>{item.years}</p>
            </div>
            <div className="case-main">
              <p className="kicker">{item.kicker}</p>
              <h3>{item.role}</h3>
              <p className="org">{item.org}</p>
              <p className="place">{item.place}</p>

              {item.id === "reu" && (
                <div className="certificate-grid" aria-label="REU certificate photo placeholders">
                  <div className="photo-placeholder"><span>PHOTO 01</span><strong>Certificate moment</strong><small>Drop your personal photo here</small></div>
                  <div className="photo-placeholder second"><span>PHOTO 02</span><strong>REU completion</strong><small>Drop your personal photo here</small></div>
                </div>
              )}

              {item.placeholderLabel && (
                <div className="image-placeholder" role="img" aria-label={`Placeholder for ${item.placeholderLabel}`}>
                  <span>IMAGE PLACEHOLDER</span>
                  <strong>{item.placeholderLabel}</strong>
                  <small>Replace with your image</small>
                </div>
              )}

              <div className="case-content">
                <ul>
                  {item.facts.map((fact) => <li key={fact}>{fact}</li>)}
                </ul>
                <aside className="metric"><strong>{item.metric.big}</strong><span>{item.metric.label}</span></aside>
              </div>

              <div className="case-footer">
                <div className="tags">{item.tools.map((tool) => <span key={tool}>{tool}</span>)}</div>
                {item.links.length > 0 && <div className="project-links">{item.links.map((link) => <a href={link.href} target="_blank" rel="noreferrer" key={link.label}>{link.label} <Arrow /></a>)}</div>}
              </div>
            </div>
          </article>
        ))}
      </section>

      <footer id="contact">
        <p className="eyebrow">CONTACT</p>
        <h2>Let’s talk.</h2>
        <a href="mailto:hello@example.com">hello@example.com <Arrow /></a>
        <p className="fineprint">Portfolio · 2026</p>
      </footer>
    </main>
  );
}