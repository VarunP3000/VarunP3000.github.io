import Link from "next/link";

export default function Page() {
  return (
    <section className="py-16 sm:py-24">
      <p className="text-xs uppercase tracking-widest text-zinc-500 dark:text-zinc-400">Welcome</p>
      <h1 className="mt-2 text-4xl sm:text-5xl lg:text-6xl font-semibold leading-tight">
        Hi, I’m Varun — I build end-to-end <span className="opacity-90">data & ML systems</span> that turn messy data into decisions.
      </h1>

      <p className="mt-5 max-w-2xl text-zinc-700 dark:text-zinc-300">
        UW Informatics + ACMS (Data Science & Statistics). I focus on forecasting, confidence scoring,
        and productionized analytics — from pipelines to dashboards.
      </p>

      <div className="mt-7 flex flex-wrap gap-3">
        <Link
          href="/projects"
          className="rounded-2xl bg-foreground/90 text-background px-5 py-3 text-sm font-medium hover:opacity-90 transition"
        >
          Explore Projects
        </Link>
        <a
          href="/Varun_Panuganti_OG_Resume.pdf"
          className="rounded-2xl border border-foreground/20 px-5 py-3 text-sm font-medium hover:bg-foreground/5 transition"
        >
          View Résumé
        </a>
      </div>

      <ul className="mt-8 flex flex-wrap items-center gap-3 text-xs text-zinc-500 dark:text-zinc-400">
        <li>Machine Learning</li>
        <li className="h-1 w-1 rounded-full bg-zinc-400/70" />
        <li>Data Visualization</li>
        <li className="h-1 w-1 rounded-full bg-zinc-400/70" />
        <li>Full-Stack (FastAPI • Next.js)</li>
      </ul>

      {/* Accent block */}
      <div className="mt-14 rounded-3xl border border-zinc-200 dark:border-zinc-800 p-6 bg-white dark:bg-zinc-950">
        <h2 className="text-lg font-semibold">Currently</h2>
        <p className="mt-2 text-sm text-zinc-700 dark:text-zinc-300">
          Building an LLM ensemble confidence scoring tool and a healthcare forecasting dashboard (readmission risk, LSTM COVID/flu, med demand).
        </p>
      </div>
    </section>
  );
}
