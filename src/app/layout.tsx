import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "../app/globals.css";
import Link from "next/link";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  metadataBase: new URL("https://VarunP3000.github.io"),
  title: { default: "Varun Panuganti – Portfolio", template: "%s • Varun Panuganti" },
  description: "Projects in DS/ML, algorithms, and statistical computing.",
  openGraph: {
    title: "Varun Panuganti – Portfolio",
    description: "Projects in DS/ML, algorithms, and statistical computing.",
    url: "/",
    siteName: "Varun Panuganti – Portfolio",
    images: ["/og.png"],
  },
  icons: { icon: "/favicon.ico" },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased bg-[--background] text-[--foreground]`}>
        {/* Sticky Nav */}
        <header className="sticky top-0 z-40 w-full border-b border-zinc-200/60 dark:border-zinc-800/60 bg-white/80 dark:bg-zinc-950/60 backdrop-blur">
          <div className="mx-auto max-w-6xl h-16 px-4 flex items-center justify-between">
            <Link href="/" className="font-semibold tracking-tight">Varun<span className="opacity-60">.ai</span></Link>
            <nav className="flex items-center gap-1">
              <Link href="/" className="px-3 py-2 rounded-xl text-sm hover:bg-zinc-100 dark:hover:bg-zinc-900">Welcome</Link>
              <Link href="/projects" className="px-3 py-2 rounded-xl text-sm hover:bg-zinc-100 dark:hover:bg-zinc-900">Projects</Link>
              <a href="/Varun_Panuganti_OG_Resume.pdf" className="ml-1 px-3 py-2 rounded-xl text-sm border border-zinc-200 dark:border-zinc-800 hover:bg-zinc-100 dark:hover:bg-zinc-900">
                Résumé
              </a>
            </nav>
          </div>
          <div className="h-px w-full bg-gradient-to-r from-transparent via-zinc-300/50 dark:via-zinc-700/50 to-transparent" />
        </header>

        {/* Page */}
        <main className="mx-auto max-w-6xl px-4">{children}</main>

        {/* Footer */}
        <footer className="mt-16 border-t border-zinc-200/60 dark:border-zinc-800/60">
          <div className="mx-auto max-w-6xl px-4 py-10 text-sm text-zinc-600 dark:text-zinc-400">
            © {new Date().getFullYear()} Varun Panuganti
          </div>
        </footer>
      </body>
    </html>
  );
}
