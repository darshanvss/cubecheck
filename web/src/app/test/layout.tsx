import { notFound } from "next/navigation";

// /test needs the test renders in public/check/, which only exist on a developer's machine.
export default function TestLayout({ children }: { children: React.ReactNode }) {
  if (process.env.NODE_ENV === "production") notFound();
  return children;
}
