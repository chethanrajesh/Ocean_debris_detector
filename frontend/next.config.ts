import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  typescript: { ignoreBuildErrors: true },
  devIndicators: false,  // hides the Route/Bundler/Turbopack popup in dev mode
};


export default nextConfig;
