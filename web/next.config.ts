import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The ONNX models change only when they're re-exported; let browsers keep them a day, then re-check.
  async headers() {
    return [{ source: "/models/:path*", headers: [{ key: "Cache-Control", value: "public, max-age=86400, must-revalidate" }] }];
  },
  async redirects() {
    return [{ source: "/support/:path*", destination: "https://cubing.in/support", permanent: true }];
  },
};

export default nextConfig;
