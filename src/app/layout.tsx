import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import Link from "next/link";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  metadataBase: new URL("https://VarunP3000.github.io"),
  title: {
    default: "Varun Panuganti – Portfolio",
    template: "%s • Varun Panuganti",
  },
  description: "Projects in DS/ML, algorithms, and statistical computing.",
  openGraph: {
    title: "Varun Panuganti – Portfolio",
    description: "Projects in DS/ML, algorithms, and statistical computing.",
    url: "/",
    siteName: "Varun Panuganti – Portfolio",
    images: ["/og.png"],
  },
  icons: {
    icon: "/favicon.ico",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-[--background] text-[--foreground]`}
      >
        {/* Sticky Nav */}
        <header className="sticky top-0 z-50 border-b border-zinc-200/60 bg-white/80 backdrop-blur dark:border-zinc-800/60 dark:bg-zinc-950/80">
          <nav className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
            <Link
              href="/"
              className="text-sm font-semibold tracking-tight"
            >
              Home Page
            </Link>

            <div className="flex items-center gap-1">
              <Link
                href="/"
                className="rounded-xl px-3 py-2 text-sm hover:bg-zinc-100 dark:hover:bg-zinc-900"
              >
                Welcome
              </Link>

              <Link
                href="/projects"
                className="rounded-xl px-3 py-2 text-sm hover:bg-zinc-100 dark:hover:bg-zinc-900"
              >
                Projects
              </Link>

              <Link
                href="/personal"
                className="rounded-xl px-3 py-2 text-sm hover:bg-zinc-100 dark:hover:bg-zinc-900"
              >
                Personal
              </Link>
            </div>
          </nav>
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