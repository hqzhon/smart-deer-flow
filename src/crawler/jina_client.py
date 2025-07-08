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
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        """关闭会话以释放连接池资源"""
        if hasattr(self, 'session'):
            self.session.close()
    
    @contextmanager
    def _get_session(self) -> Generator[requests.Session, None, None]:
        """上下文管理器确保会话正确关闭"""
        try:
            yield self.session
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
    
    def crawl(self, url: str, return_format: str = "html") -> str:
        headers = {
            "Content-Type": "application/json",
            "X-Return-Format": return_format,
        }
        if os.getenv("JINA_API_KEY"):
            headers["Authorization"] = f"Bearer {os.getenv('JINA_API_KEY')}"
        else:
            logger.warning(
                "Jina API key is not set. Provide your own key to access a higher rate limit. See https://jina.ai/reader for more information."
            )
        data = {"url": url}
        
        try:
            with self._get_session() as session:
                response = session.post("https://r.jina.ai/", headers=headers, json=data, timeout=30)
                response.raise_for_status()  # 检查HTTP错误
                return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to crawl URL {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while crawling URL {url}: {e}")
            raise
