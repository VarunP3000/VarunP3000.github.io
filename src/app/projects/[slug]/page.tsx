import { notFound } from "next/navigation";
import { projects } from "../../../lib/projects";

export const dynamic = "error"; // ensure static only

type Props = { params: { slug: string } };

export function generateStaticParams() {
  return projects.map((p) => ({ slug: p.slug }));
}

export function generateMetadata({ params }: Props) {
  const p = projects.find((x) => x.slug === params.slug);
  return {
    title: p ? p.title : "Project",
    description: p?.summary ?? "Project case study",
  };
}

export default function ProjectCaseStudy({ params }: Props) {
  const p = projects.find((x) => x.slug === params.slug);
  if (!p) return notFound();

  return (
    <article className="py-12 sm:py-16">
      <header className="mb-8">
        <p className="text-xs uppercase tracking-widest text-zinc-500 dark:text-zinc-400">Case Study</p>
        <h1 className="mt-2 text-3xl sm:text-4xl font-semibold">{p.title}</h1>

        <div className="mt-3 flex flex-wrap items-center gap-3 text-sm text-zinc-600 dark:text-zinc-400">
          {p.year && <span>{p.year}</span>}
          {p.tags?.length ? (
            <>
              <span className="h-1 w-1 rounded-full bg-zinc-400/60" />
              <ul className="flex flex-wrap gap-2">
                {p.tags.map((t) => (
                  <li key={t} className="rounded-full border border-zinc-200 dark:border-zinc-800 px-2 py-0.5 text-xs">
                    {t}
                  </li>
                ))}
              </ul>
            </>
          ) : null}
        </div>

        <p className="mt-4 max-w-prose text-zinc-700 dark:text-zinc-300">{p.summary}</p>

        <div className="mt-5 flex gap-3 text-sm">
          {p.repo && (
            <a
              href={p.repo}
              target="_blank"
              className="rounded-xl border border-zinc-200 dark:border-zinc-800 px-3 py-2 hover:bg-zinc-100 dark:hover:bg-zinc-900"
            >
              GitHub
            </a>
          )}
          {p.live && (
            <a
              href={p.live}
              target="_blank"
              className="rounded-xl border border-zinc-200 dark:border-zinc-800 px-3 py-2 hover:bg-zinc-100 dark:hover:bg-zinc-900"
            >
              Live Demo
            </a>
          )}
        </div>
      </header>

      <section className="prose prose-zinc dark:prose-invert max-w-none">
        {p.problem && (
          <>
            <h2 className="text-lg font-bold mb-1">Problem & Motivation:</h2>
            <p>{p.problem}</p>
          </>
        )}

        {p.approach && (
          <>
            <h2 className="text-lg font-bold mb-1">Data & Approach:</h2>
            {Array.isArray(p.approach) ? (
              <ul>{p.approach.map((a, i) => <li key={i}>{a}</li>)}</ul>
            ) : (
              <p>{p.approach}</p>
            )}
          </>
        )}

        {p.results && (
          <>
            <h2 className="text-lg font-bold mb-1">Results:</h2>
            {Array.isArray(p.results) ? (
              <ul>{p.results.map((r, i) => <li key={i}>{r}</li>)}</ul>
            ) : (
              <p>{p.results}</p>
            )}
          </>
        )}

        {p.limitations && (
          <>
            <h2 className="text-lg font-bold mb-1">Limitations:</h2>
            <p>{p.limitations}</p>
          </>
        )}

        {p.nextSteps && (
          <>
            <h2 className="text-lg font-bold mb-1">Next Steps:</h2>
            {Array.isArray(p.nextSteps) ? (
              <ul>{p.nextSteps.map((n, i) => <li key={i}>{n}</li>)}</ul>
            ) : (
              <p>{p.nextSteps}</p>
            )}
          </>
        )}
      </section>
    </article>
  );
}
