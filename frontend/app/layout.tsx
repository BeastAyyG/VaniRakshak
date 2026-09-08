import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "VaniRakshak — Enterprise Audio Deepfake & Voice Forensics",
  description:
    "Target-conditioned recorded-audio deepfake & voice cloning forensic intelligence. Segment-level risk localization, ECAPA-TDNN speaker verification, and DPDP Act 2023 compliance.",
  icons: {
    icon: [
      { url: "/brand/vani-rakshak-icon.jpg", type: "image/jpeg" },
    ],
    shortcut: ["/brand/vani-rakshak-icon.jpg"],
    apple: [{ url: "/brand/vani-rakshak-icon.jpg", type: "image/jpeg" }],
  },
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
