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
      "Co-authored a peer-reviewed paper accepted for publication at EUSPN 2026, benchmarking 5 emotion-annotated text datasets across social media, dialogue, and educational domains to study emotion representation and cross-domain generalization.",
      "Designed and executed a large-scale experimental pipeline comparing classical machine learning and transformer models across multiple datasets, using repeated grouped cross-validation and standardized evaluation metrics to ensure reproducible results.",
      "Built scalable NLP and data-processing workflows for 15M+ text samples, combining distributed GPU inference, ensemble modeling, and statistical analysis to evaluate model performance, efficiency, and label behavior.",
    ],
    links: [
      { label: "Research repository", href: "https://github.com/VarunP3000/NFS-REU-Data-Analytics-2026" },
      { label: "2026 REU program", href: "https://www.marshall.edu/reu/2026-gallery/" },
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
            <a className="text-link" href="varunpanu30@gmail.com">Get in touch <Arrow /></a>
          </div>
        </div>
        <div className="hero-photo">
          <img
            src="/headshot.jpg"
            alt="Portrait of Varun"
            className="hero-image"
          />
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
                <div className="certificate-grid">
                  <img
                    src="/REUCertificatePhoto.png"
                    alt="Varun holding his REU certificate"
                    className="certificate-image"
                  />
                  <img
                    src="/REUGroupPic.png"
                    alt="REU group photo"
                    className="certificate-image"
                  />
                </div>
              )}

              {item.id === "ta" && (
                <div className="experience-image">
                  <img
                    src="/UWCampus.png"
                    alt="University of Washington campus"
                    className="experience-image-file campus-image"
                  />
                </div>
              )}

              {item.id === "icode" && (
                <div className="experience-image">
                  <img
                    src="/iCode_Logo.jpg"
                    alt="iCode logo"
                    className="experience-image-file logo-image"
                  />
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
        <a href="mailto:varunpanu30@gmail.com">
          varunpanu30@gmail.com <Arrow />
        </a>
        <p className="fineprint">Portfolio · 2026</p>
      </footer>
    </main>
  );
}