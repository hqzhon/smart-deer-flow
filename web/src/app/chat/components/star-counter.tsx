"use client";

import { StarFilledIcon } from "@radix-ui/react-icons";
import { useEffect, useState } from "react";

import { NumberTicker } from "~/components/magicui/number-ticker";
import { env } from "~/env";

export function StarCounter() {
  const [stars, setStars] = useState(1000);

  useEffect(() => {
    const fetchStars = async () => {
      try {
        const response = await fetch(
          "https://api.github.com/repos/bytedance/deer-flow",
          {
            headers: env.GITHUB_OAUTH_TOKEN
              ? {
                  Authorization: `Bearer ${env.GITHUB_OAUTH_TOKEN}`,
                  "Content-Type": "application/json",
                }
              : {},
          },
        );

        if (response.ok) {
          const data = await response.json();
          setStars(data.stargazers_count ?? 1000);
        }
      } catch (error) {
        console.error("Error fetching GitHub stars:", error);
      }
    };

    void fetchStars();
  }, []);

  return (
    <>
      <StarFilledIcon className="size-4 transition-colors duration-300 group-hover:text-yellow-500" />
      <NumberTicker className="font-mono tabular-nums" value={stars} />
    </>
  );
}