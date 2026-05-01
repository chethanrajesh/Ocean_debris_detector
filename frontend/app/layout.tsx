import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Ocean Debris Monitor — Global Plastic Tracking System",
  description:
    "Research-grade global ocean plastic debris monitoring and prediction dashboard. " +
    "Powered by Lagrangian particle physics + Graph Neural Networks trained on 30 years " +
    "of NOAA GDP buoy data. Data sources: NASA Earthdata, NOAA ERDDAP, CMEMS, Sentinel-2.",
  keywords: [
    "ocean plastic",
    "debris monitoring",
    "machine learning",
    "ocean currents",
    "environmental research",
    "GNN",
    "Lagrangian simulation",
  ],
  authors: [{ name: "Ocean Debris Research Team" }],
  robots: "index, follow",
  openGraph: {
    title: "Ocean Debris Monitor",
    description: "Global ocean plastic debris tracking and prediction system",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        {/* Google Earth Engine JavaScript API */}
        <script
          src="https://maps.googleapis.com/maps/api/js?key=AIzaSyAAR0xfDfY3yLQtyvSNTCeQKnd91ea-8fQ&v=weekly&loading=async&libraries=marker"
        />
      </head>
      <body className="bg-[#020a12] overflow-hidden">
        {children}
      </body>
    </html>
  );
}
