import { projects } from "../../lib/projects";
import { ProjectCard } from "../../components/ProjectCard";

export const metadata = {
  title: "Projects",
};

export default function ProjectsPage() {
  return (
    <section className="py-16 sm:py-20">
      <header className="mb-8">
        <h1 className="text-3xl sm:text-4xl font-semibold">Projects</h1>
        <p className="mt-2 max-w-prose text-zinc-700 dark:text-zinc-300">
        Here are my projects:
        </p>
      </header>

      <div className="grid gap-6 sm:grid-cols-2">
        {projects.map((p) => (
          <ProjectCard key={p.slug} {...p} />
        ))}
      </div>
    </section>
  );
}
