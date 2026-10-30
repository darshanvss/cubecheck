import Link from "next/link";
import { CubeLogo } from "@/components/cube-logo";
import { ThemeToggle } from "@/components/theme-toggle";

export function SiteHeader() {
  return (
    <header className="border-b border-border bg-background/80">
      <nav aria-label="Main" className="mx-auto flex max-w-[1536px] items-center justify-between gap-3 px-5 py-3 lg:px-8">
        <Link href="/" className="flex shrink-0 items-center gap-2 text-lg font-semibold tracking-tight">
          <CubeLogo />
          CubeCheck
        </Link>
        <div className="flex items-center gap-1 text-sm text-muted-foreground sm:gap-2">
          <Link href="/about" className="whitespace-nowrap rounded-lg px-3 py-2 hover:text-foreground">How it works</Link>
          <ThemeToggle />
        </div>
      </nav>
    </header>
  );
}
