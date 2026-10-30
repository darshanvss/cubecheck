import type { Metadata } from "next";

export const metadata: Metadata = { title: "How it works" };

const steps = [
  ["Find the cube", "A small network looks at the whole frame and answers whether there's a cube, and roughly where its 7 visible corners are."],
  ["Zoom in", "A second network looks at a close-up of that spot in the full-quality frame and places the corners exactly. In live video it follows the cube from frame to frame."],
  ["Read the stickers", "The corners give a 3 × 3 grid on each of the three faces. A third network reads all 27 stickers together: one of the six colours, or covered by a finger."],
  ["Check the scramble", "Each scramble gives the cube you should be holding. The reading is compared with it, trying all 24 ways the cube could be held, and with every version of the scramble with one move slipped. If a slip fits clearly better, it's a mis-scramble."],
];

export default function About() {
  return (
    <div className="mx-auto w-full max-w-2xl px-5 py-10 sm:py-14">
      <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">How it works</h1>
      <p className="mt-3 leading-relaxed text-muted-foreground">
        Hold the cube with one corner toward the camera and three faces show: 27 of its 54 stickers. That&rsquo;s enough, because a slipped move changes about 8 of the 24 stickers you can see.
      </p>
      <ol className="mt-8 space-y-6">
        {steps.map(([title, text], i) => (
          <li key={title} className="flex gap-4">
            <span className="grid size-8 shrink-0 place-items-center rounded-full bg-primary/10 text-sm font-semibold text-primary">{i + 1}</span>
            <div>
              <h2 className="font-semibold">{title}</h2>
              <p className="mt-1 text-sm leading-relaxed text-muted-foreground">{text}</p>
            </div>
          </li>
        ))}
      </ol>
      <h2 className="mt-10 font-semibold">Private by design</h2>
      <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
        The three networks run in your browser, on your device&rsquo;s GPU where it has one. Camera frames and scrambles never leave your device, which also keeps competition scrambles secret.
      </p>
      <h2 className="mt-8 font-semibold">Trained on renders only</h2>
      <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
        The networks learned from tens of thousands of 3D renders of cubes in rooms, in hands, under all kinds of light, with exact labels. On held-out renders the check calls the true scramble correct 99% of the time with no false alarms, and catches 99% of single slipped moves. Results on real photos come next.
      </p>
      <p className="mt-8 text-sm leading-relaxed text-muted-foreground">
        CubeCheck helps the scrambler&rsquo;s own check; it isn&rsquo;t an official WCA tool. It&rsquo;s <a href="https://github.com/darshanvss/cubecheck" className="underline underline-offset-4 hover:text-foreground">open source</a> under the AGPL, and the models are on <a href="https://huggingface.co/darshanvss/cubecheck-3x3" className="underline underline-offset-4 hover:text-foreground">Hugging Face</a>.
      </p>
    </div>
  );
}
