import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

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
    images: ["/og.png"], // optional; add /public/og.png if you want previews
  },
  icons: { icon: "/favicon.ico" }, // optional; add /public/favicon.ico
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        {children}
      </body>
    </html>
  );
}
