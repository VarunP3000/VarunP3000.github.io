"use client";

import { useMemo, useState } from "react";
import { projects } from "../../../lib/projects";
import { ProjectCard } from "../../../components/ProjectCard";

export default function AllProjectsPage() {
  const [active, setActive] = useState<string>("All");

  const allTags = useMemo(() => {
    const s = new Set<string>();
    projects.forEach((p) => p.tags.forEach((t) => s.add(t)));
    return ["All", ...Array.from(s).sort((a, b) => a.localeCompare(b))];
  }, []);

  const filtered = useMemo(() => {
    if (active === "All") return projects;
    return projects.filter((p) => p.tags.includes(active));
  }, [active]);

  return (
    <section className="py-16 sm:py-20">
      <header className="mb-8">
        <h1 className="text-3xl sm:text-4xl font-semibold">All Projects</h1>
        <p className="mt-2 max-w-prose text-zinc-700 dark:text-zinc-300">
          Filter by tag to find specific work.
        </p>
      </header>

      <div className="mb-8 flex flex-wrap gap-2">
        {allTags.map((t) => {
          const selected = t === active;
          return (
            <button
              key={t}
              onClick={() => setActive(t)}
              className={[
                "rounded-full px-3 py-1.5 text-xs border transition",
                selected
                  ? "bg-foreground text-background border-foreground"
                  : "border-zinc-200 dark:border-zinc-800 hover:bg-zinc-100 dark:hover:bg-zinc-900",
              ].join(" ")}
            >
              {t}
            </button>
          );
        })}
      </div>

      <div className="grid gap-6 sm:grid-cols-2">
        {filtered.map((p) => (
          <ProjectCard key={p.slug} {...p} />
        ))}
      </div>
    </section>
  );
}