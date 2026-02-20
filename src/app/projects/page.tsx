import Link from "next/link";
import { projects } from "../../lib/projects";
import { ProjectCard } from "../../components/ProjectCard";

export const metadata = {
  title: "Selected Work",
};

export default function ProjectsPage() {
  const featured = projects.filter((p) => p.featured);

  return (
    <section className="py-16 sm:py-20">
      <header className="mb-8">
        <h1 className="text-3xl sm:text-4xl font-semibold">Selected Work</h1>
        <p className="mt-2 max-w-prose text-zinc-700 dark:text-zinc-300">
          A curated set of projects I can go deep on.
        </p>
      </header>

      <div className="grid gap-6 sm:grid-cols-2">
        {featured.map((p) => (
          <ProjectCard key={p.slug} {...p} />
        ))}
      </div>

      <div className="mt-10 flex justify-center">
        <Link
          href="/projects/all"
          className="rounded-2xl border border-zinc-200 dark:border-zinc-800 px-5 py-3 text-sm font-medium hover:bg-zinc-100 dark:hover:bg-zinc-900 transition"
        >
          View all projects →
        </Link>
      </div>
    </section>
  );
}
