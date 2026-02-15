import Image from "next/image";
import Link from "next/link";

export default function Page() {
  return (
    <section className="py-16 sm:py-24">
      <p className="text-xs uppercase tracking-widest text-zinc-500 dark:text-zinc-400">
        Welcome
      </p>
      <h1 className="mt-2 text-4xl sm:text-5xl lg:text-6xl font-semibold leading-tight">
        About Me:
      </h1>

      <div className="mt-6 flex flex-col gap-6 sm:flex-row sm:items-start">
        {/* Text block */}
        <div className="max-w-2xl text-zinc-700 dark:text-zinc-300">
          <p>
            My name is Varun Panuganti. I am a third year student at the
            University of Washington pursuing a double degree in ACMS (Data
            Science and Statistics) and Informatics. I am passionate about
            studying how data and mathematics can be used to make decisions. I
            am also driven to use my privilege and my experience to build
            accessible and informative products.
          </p>
        </div>

        {/* Image block */}
        <div className="shrink-0">
          <Image
            src="/headshot.jpg" // file in /public
            alt="Portrait of Varun Panuganti"
            width={220}
            height={220}
            className="rounded-2xl object-cover border border-zinc-200 dark:border-zinc-800"
          />
        </div>
      </div>

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
          View Resume
        </a>

        <a
          href="/transcript.pdf"
          target="_blank"
          rel="noopener noreferrer"
          className="rounded-2xl border border-foreground/20 px-5 py-3 text-sm font-medium hover:bg-foreground/5 transition"
        >
          Transcript
        </a>

        <a
          href="https://github.com/VarunP3000"
          target="_blank"
          rel="noopener noreferrer"
          className="rounded-2xl border border-foreground/20 px-5 py-3 text-sm font-medium hover:bg-foreground/5 transition"
        >
          GitHub
        </a>

        <a
          href="https://www.linkedin.com/in/varun-panuganti/"
          target="_blank"
          rel="noopener noreferrer"
          className="rounded-2xl border border-foreground/20 px-5 py-3 text-sm font-medium hover:bg-foreground/5 transition"
        >
          LinkedIn
        </a>
      </div>

      <ul className="mt-8 flex flex-wrap items-center gap-3 text-xs text-zinc-500 dark:text-zinc-400">
        <li>Machine Learning</li>
        <li className="h-1 w-1 rounded-full bg-zinc-400/70" />
        <li>Data Visualization</li>
        <li className="h-1 w-1 rounded-full bg-zinc-400/70" />
        <li>Full-Stack (FastAPI • Next.js)</li>
      </ul>
    </section>
  );
}
