import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VQE Explorer",
  description: "Interactive variational quantum eigensolver simulations",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
