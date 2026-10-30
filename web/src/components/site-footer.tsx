export function SiteFooter() {
  return (
    <footer className="mt-auto border-t border-border bg-background/80">
      <div className="mx-auto flex max-w-[1536px] flex-col items-center justify-between gap-4 px-5 py-6 lg:px-8 text-sm text-muted-foreground sm:flex-row">
        <div><span className="font-medium text-foreground">CubeCheck</span> © {new Date().getFullYear()}</div>
        <nav aria-label="Footer" className="flex flex-wrap justify-center gap-5">
          <a href="https://cubing.in/support">Support</a>
          <a href="https://github.com/darshanvss/cubecheck">GitHub</a>
          <a href="https://cubing.in">cubing.in</a>
        </nav>
      </div>
    </footer>
  );
}
