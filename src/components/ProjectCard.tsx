import Link from "next/link";

export type Project = {
  slug: string;
  title: string;
  summary: string;
  tags: string[];
  year?: string | number;
  repo?: string;
  live?: string;
};

export function ProjectCard({ slug, title, summary, tags, year, repo, live }: Project) {
  return (
    <article className="group relative overflow-hidden rounded-3xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 p-5">
      <div className="flex items-start justify-between gap-4">
        <h3 className="text-lg font-semibold">{title}</h3>
        {year && <span className="text-xs text-zinc-500 dark:text-zinc-400">{year}</span>}
      </div>

      <p className="mt-2 line-clamp-4 text-sm text-zinc-700 dark:text-zinc-300">{summary}</p>

      <div className="mt-3 flex flex-wrap gap-2">
        {tags.map((t) => (
          <span key={t} className="rounded-full border border-zinc-200 dark:border-zinc-800 px-2 py-1 text-[11px]">
            {t}
          </span>
        ))}
      </div>

      <div className="mt-4 flex gap-3 text-sm">
        <Link href={`/projects/${slug}`} className="rounded-xl bg-foreground/90 text-background px-3 py-2">
          Case Study
        </Link>
        {live && (
          <a href={live} target="_blank" className="rounded-xl border border-zinc-200 dark:border-zinc-800 px-3 py-2 hover:bg-zinc-100 dark:hover:bg-zinc-900">
            Live
          </a>
        )}
        {repo && (
          <a href={repo} target="_blank" className="rounded-xl border border-zinc-200 dark:border-zinc-800 px-3 py-2 hover:bg-zinc-100 dark:hover:bg-zinc-900">
            GitHub
          </a>
        )}
      </div>
    </article>
  );
}
