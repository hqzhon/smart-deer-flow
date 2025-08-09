# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
import os
from contextlib import contextmanager
from typing import Generator

import requests

logger = logging.getLogger(__name__)


class JinaClient:
    def __init__(self):
        self.session = requests.Session()
        # 设置连接池参数以避免资源泄漏
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10, pool_maxsize=20, max_retries=3
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """关闭会话以释放连接池资源"""
        if hasattr(self, "session"):
            self.session.close()

    @contextmanager
    def _get_session(self) -> Generator[requests.Session, None, None]:
        """上下文管理器确保会话正确关闭"""
        try:
            yield self.session
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise

    def _direct_fetch(self, url: str) -> str:
        """Directly fetch the page content as a fallback when Jina API is unavailable.

        This uses a standard GET request with a reasonable User-Agent to reduce the
        chance of being blocked by the target site.
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        with self._get_session() as session:
            resp = session.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.text

    def crawl(self, url: str, return_format: str = "html") -> str:
        headers = {
            "Content-Type": "application/json",
            "X-Return-Format": return_format,
        }
        if os.getenv("JINA_API_KEY"):
            headers["Authorization"] = f"Bearer {os.getenv('JINA_API_KEY')}"
        else:
            logger.warning(
                "Jina API key is not set. Provide your own key to access a higher rate limit. "
                "See https://jina.ai/reader for more information."
            )
        data = {"url": url}

        try:
            with self._get_session() as session:
                response = session.post(
                    "https://r.jina.ai/", headers=headers, json=data, timeout=30
                )
                try:
                    response.raise_for_status()  # Check HTTP errors
                    return response.text
                except requests.exceptions.HTTPError as http_err:
                    status = getattr(response, "status_code", None)
                    logger.error(f"Jina API error (status={status}): {http_err}")
                    # Gracefully fall back for common rate/plan errors
                    if status in {401, 402, 403, 429, 503}:
                        logger.warning(
                            "Falling back to direct fetch due to Jina API unavailability."
                        )
                        return self._direct_fetch(url)
                    raise
        except requests.exceptions.RequestException as e:
            # Network-related error, fall back to direct fetch
            logger.error(f"Failed to crawl via Jina, falling back to direct fetch: {e}")
            return self._direct_fetch(url)
        except Exception as e:
            logger.error(f"Unexpected error while crawling URL {url}: {e}")
            # Best-effort fallback
            return self._direct_fetch(url)
