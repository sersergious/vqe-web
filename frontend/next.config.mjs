/** @type {import('next').NextConfig} */
const nextConfig = {
  // Production builds emit plain static files into out/, which FastAPI serves.
  // Left off in dev so the rewrite below works (rewrites are ignored on export).
  output: process.env.NODE_ENV === "production" ? "export" : undefined,

  // Dev only: in production the API and the page share an origin, so /api/*
  // already lands on FastAPI without any rewriting.
  async rewrites() {
    const backend = process.env.BACKEND_URL ?? "http://localhost:8000";
    return [{ source: "/api/:path*", destination: `${backend}/api/:path*` }];
  },
};

export default nextConfig;
