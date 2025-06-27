"use client";

import { GitHubLogoIcon } from "@radix-ui/react-icons";
import dynamic from "next/dynamic";
import Link from "next/link";

import { Button } from "~/components/ui/button";
import { env } from "~/env";

const StarCounter = dynamic(() => import("./star-counter").then(mod => mod.StarCounter), {
  ssr: true,
  loading: () => (
    <>
      <div className="size-4" />
      <span className="font-mono tabular-nums">1000</span>
    </>
  )
});

export function GitHubButton() {
  return (
    <Button
      variant="outline"
      size="sm"
      asChild
      className="group relative z-10"
    >
      <Link 
        href="https://github.com/bytedance/deer-flow" 
        target="_blank"
        className="inline-flex items-center justify-center gap-2"
      >
        <GitHubLogoIcon className="size-4" />
        Star on GitHub
        {env.NEXT_PUBLIC_STATIC_WEBSITE_ONLY &&
          env.GITHUB_OAUTH_TOKEN && <StarCounter />}
      </Link>
    </Button>
  );
}