// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT
"use client";

import { GitHubLogoIcon } from "@radix-ui/react-icons";
import dynamic from "next/dynamic";
import Link from "next/link";

import { Button } from "~/components/ui/button";
import { env } from "~/env";

const StarCounter = dynamic(() => import("./star-counter").then(mod => mod.StarCounter), {
  ssr: false
});

export function SiteHeader() {
  return (
    <header className="supports-backdrop-blur:bg-background/80 bg-background/40 sticky top-0 left-0 z-40 flex h-15 w-full flex-col items-center backdrop-blur-lg">
      <div className="container flex h-15 items-center justify-between px-3">
        <div className="text-xl font-medium">
          <span className="mr-1 text-2xl">🦌</span>
          <span>DeerFlow</span>
        </div>
        <div className="relative flex items-center">
          <div
            className="pointer-events-none absolute inset-0 z-0 h-full w-full rounded-full opacity-60 blur-2xl"
            style={{
              background: "linear-gradient(90deg, #ff80b5 0%, #9089fc 100%)",
              filter: "blur(32px)",
            }}
          />
          <Button
            variant="outline"
            size="sm"
            asChild
            className="group relative z-10"
          >
            <Link href="https://github.com/bytedance/deer-flow" target="_blank">
              <GitHubLogoIcon className="size-4" />
              Star on GitHub
              {env.NEXT_PUBLIC_STATIC_WEBSITE_ONLY &&
                env.GITHUB_OAUTH_TOKEN && <StarCounter />}
            </Link>
          </Button>
        </div>
      </div>
      <hr className="from-border/0 via-border/70 to-border/0 m-0 h-px w-full border-none bg-gradient-to-r" />
    </header>
  );
}
