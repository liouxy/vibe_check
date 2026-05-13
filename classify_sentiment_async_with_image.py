#!/usr/bin/env python3
"""
评论情感分类工具（异步版，支持图片 URL）
支持批量处理 CSV 中的评论，调用 OpenAI API 进行情感分类，并支持断点续传。

实现要点：
1. 多 worker 并发请求模型
2. 单 writer 串行写入 cache，保留原有 cache 追加语义
3. 支持从 CSV 的图片 URL 列读取 URL 数组，并放入 user input
"""

import argparse
import asyncio
import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd


try:
    from openai import AsyncOpenAI
except ImportError:
    print("请先安装依赖: pip install openai")
    raise SystemExit(1)


def extract_json_from_llm(text: str):
    """不管 AI 是否带 ```json，都能准确把字典提取出来。"""
    markdown_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(markdown_pattern, text)

    if match:
        json_str = match.group(1).strip()
    else:
        brace_pattern = r"(\{[\s\S]*\})"
        match = re.search(brace_pattern, text)
        if match:
            json_str = match.group(1).strip()
        else:
            json_str = text.strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print(f"解析失败！原始文本: {text}")
        return None


class AsyncSentimentClassifier:
    """异步情感分类器。"""

    CACHE_VARIANT = "image_input"

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        system_prompt_path: str = "prompts/nintendo_comment_classify.txt",
        user_prompt_prefix: str = "",
        cache_dir: str = "cache",
    ):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.system_prompt = self._load_system_prompt(system_prompt_path)
        self.user_prompt_prefix = user_prompt_prefix.strip()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _load_system_prompt(self, prompt_path: str) -> str:
        prompt_file = Path(prompt_path)
        if not prompt_file.exists():
            raise FileNotFoundError(f"系统提示词文件不存在: {prompt_path}")

        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _get_cache_path(self, input_file: str) -> Path:
        input_name = Path(input_file).stem
        cache_key = f"{self.CACHE_VARIANT}\n{self.user_prompt_prefix}"
        if not self.user_prompt_prefix:
            variant_hash = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:8]
            return self.cache_dir / f"{input_name}_{variant_hash}_cache.jsonl"

        prefix_hash = hashlib.md5(
            cache_key.encode("utf-8")
        ).hexdigest()[:8]
        return self.cache_dir / f"{input_name}_{prefix_hash}_cache.jsonl"

    def _parse_image_urls(self, raw_value: str) -> List[str]:
        if not raw_value or not raw_value.strip():
            return []

        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "image_url 列必须是 JSON 数组字符串，例如: "
                '["https://example.com/a.png", "https://example.com/b.png"]'
            ) from exc

        if not isinstance(parsed, list) or not all(isinstance(url, str) for url in parsed):
            raise ValueError("image_url 列必须是字符串 URL 组成的 JSON 数组")

        return parsed

    def _build_user_message(
        self,
        comment: str,
        image_urls: Optional[List[str]] = None,
    ) -> Union[str, List[Dict[str, Any]]]:
        text_content = comment
        if self.user_prompt_prefix:
            text_content = f"{self.user_prompt_prefix}\n\n{comment}"

        if not image_urls:
            return text_content

        content: List[Dict[str, Any]] = [{"type": "text", "text": text_content}]
        for url in image_urls:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": url},
                }
            )
        return content

    def _load_cache(self, cache_path: Path) -> Dict[int, Dict]:
        cache: Dict[int, Dict] = {}
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line.strip())
                    cache[item["index"]] = item
        return cache

    async def classify_comment(
        self,
        comment: str,
        image_urls: Optional[List[str]] = None,
    ) -> Dict:
        """对单条评论进行异步情感分类。"""
        try:
            user_message = self._build_user_message(comment, image_urls=image_urls)
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0,
                max_tokens=2000,
            )

            result = response.choices[0].message.content
            parsed_result = extract_json_from_llm(result)

            return {
                "success": True,
                "result": parsed_result,
                "raw_response": result,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": None,
            }

    async def process_csv(
        self,
        input_file: str,
        output_file: str,
        comment_fields: List[str],
        image_url_field: str,
        max_retries: int = 3,
        workers: int = 8,
        requests_per_second: float = 5.0,
    ):
        """异步处理 CSV 文件中的评论。"""
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_file}")

        cache_path = self._get_cache_path(input_file)
        cache = self._load_cache(cache_path)
        print(f"已加载缓存: {len(cache)} 条记录")

        with open(input_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        total = len(rows)
        print(f"总共 {total} 条评论需要处理")

        results_by_index: Dict[int, Dict] = {}
        jobs: List[Tuple[int, Dict, str, List[str]]] = []

        for idx, row in enumerate(rows):
            if idx in cache:
                print(f"[{idx + 1}/{total}] 已缓存，跳过")
                results_by_index[idx] = cache[idx]
                continue

            comment = " ".join([row.get(field, "") for field in comment_fields])


            try:
                image_urls = self._parse_image_urls(row.get(image_url_field, ""))
            except ValueError as exc:
                print(f"[{idx + 1}/{total}] 图片 URL 列格式错误，跳过: {exc}")
                results_by_index[idx] = {
                    "index": idx,
                    **{k: v for k, v in row.items()},
                    "raw_response": None,
                    "error": str(exc),
                }
                continue

            jobs.append((idx, row, comment, image_urls))

        print(f"待异步处理: {len(jobs)} 条，worker 数: {workers}")

        job_queue: asyncio.Queue = asyncio.Queue()
        write_queue: asyncio.Queue = asyncio.Queue()
        progress_lock = asyncio.Lock()
        rate_lock = asyncio.Lock()
        progress = {"done": 0, "total": len(jobs)}
        last_request_time = {"value": 0.0}

        for job in jobs:
            await job_queue.put(job)

        for _ in range(workers):
            await job_queue.put(None)

        async def writer_task():
            with open(cache_path, "a", encoding="utf-8") as f:
                while True:
                    item = await write_queue.get()
                    if item is None:
                        write_queue.task_done()
                        break
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    f.flush()
                    write_queue.task_done()

        async def worker_task(worker_id: int):
            while True:
                job = await job_queue.get()
                if job is None:
                    job_queue.task_done()
                    break

                idx, row, comment, image_urls = job
                success = False

                for retry in range(max_retries):
                    print(
                        f"[worker-{worker_id}] [{idx + 1}/{total}] 正在分类... "
                        f"(尝试 {retry + 1}/{max_retries})"
                    )

                    min_interval = 1.0 / requests_per_second
                    async with rate_lock:
                        now = time.monotonic()
                        wait_time = min_interval - (now - last_request_time["value"])
                        if wait_time > 0:
                            await asyncio.sleep(wait_time)
                        last_request_time["value"] = time.monotonic()

                    classification = await self.classify_comment(
                        comment,
                        image_urls=image_urls,
                    )

                    if classification["success"]:
                        item = {
                            "index": idx,
                            **{k: v for k, v in row.items()},
                            "raw_response": classification["raw_response"],
                        }
                        item.update(classification["result"] or {})

                        results_by_index[idx] = item
                        await write_queue.put(item)
                        success = True
                        break

                    print(f"[worker-{worker_id}] 错误: {classification['error']}")

                if not success:
                    print(f"[worker-{worker_id}] [{idx + 1}/{total}] 处理失败，跳过")
                    results_by_index[idx] = {
                        "index": idx,
                        **{k: v for k, v in row.items()},
                        "raw_response": None,
                        "error": "error!",
                    }

                async with progress_lock:
                    progress["done"] += 1
                    print(f"进度: {progress['done']}/{progress['total']}")

                job_queue.task_done()

        writer = asyncio.create_task(writer_task())
        workers_list = [
            asyncio.create_task(worker_task(i + 1))
            for i in range(workers)
        ]

        await job_queue.join()
        await asyncio.gather(*workers_list)

        await write_queue.put(None)
        await write_queue.join()
        await writer

        results: List[Dict] = []
        for idx in range(total):
            item = results_by_index.get(idx)
            if item is not None:
                results.append(item)

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False, encoding="utf-8-sig")

        print(f"\n处理完成! 结果已保存至: {output_file}")
        print(f"成功: {sum(1 for r in results if r.get('raw_response'))} 条")
        print(f"失败: {sum(1 for r in results if not r.get('raw_response'))} 条")


def main():
    parser = argparse.ArgumentParser(
        description="评论情感分类工具（异步版，支持图片 URL）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置处理单个文件
  python classify_sentiment_async_with_image.py -i inputs/comments.csv -o outputs/result.csv

  # 指定图片 URL 列
  python classify_sentiment_async_with_image.py -i inputs/comments.csv -o outputs/result.csv \
    --image-url-field image_url

  # 指定并发 worker 数
  python classify_sentiment_async_with_image.py -i inputs/comments.csv -o outputs/result.csv --workers 16
        """,
    )

    parser.add_argument("-i", "--input", type=str, required=True, help="输入CSV文件路径")
    parser.add_argument("-o", "--output", type=str, required=True, help="输出CSV文件路径")
    parser.add_argument(
        "--comment-fields",
        type=str,
        default="comment",
        help="CSV中评论字段名，多个字段用逗号分隔（默认: comment）",
    )
    parser.add_argument(
        "--image-url-field",
        type=str,
        default="image_url",
        help="CSV 中图片 URL 列名，列值需为 JSON 数组字符串（默认: image_url）",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY", ""),
        help="OpenAI API密钥（默认从环境变量 OPENAI_API_KEY 读取）",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="API端点URL（默认使用官方 endpoint）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("DEFAULT_MODEL", "gpt-4o-mini"),
        help="使用的模型（默认: gpt-4o-mini，支持环境变量 DEFAULT_MODEL）",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="prompts/nintendo_comment_classify.txt",
        help="系统提示词文件路径（默认: prompts/nintendo_comment_classify.txt）",
    )
    parser.add_argument(
        "--user-prefix",
        type=str,
        default="",
        help="追加到 user prompt 前面的上下文前缀，例如：这是用户的评论",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="最大重试次数（默认: 2）",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="cache",
        help="缓存目录（默认: cache）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="异步并发 worker 数（默认: 8）",
    )
    parser.add_argument(
        "--requests-per-second",
        type=float,
        default=5.0,
        help="全局请求速率上限（默认: 5.0 次/秒）",
    )

    args = parser.parse_args()
    print(args)

    if not args.api_key:
        print("错误: 请通过 --api-key 参数或 OPENAI_API_KEY 环境变量提供 API 密钥")
        return 1

    if args.requests_per_second <= 0:
        print("错误: --requests-per-second 必须大于 0")
        return 1

    try:
        classifier = AsyncSentimentClassifier(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            system_prompt_path=args.prompt,
            user_prompt_prefix=args.user_prefix,
            cache_dir=args.cache_dir,
        )
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return 1

    asyncio.run(
        classifier.process_csv(
            input_file=args.input,
            output_file=args.output,
            comment_fields=args.comment_fields.split(","),
            image_url_field=args.image_url_field,
            max_retries=args.max_retries,
            workers=args.workers,
            requests_per_second=args.requests_per_second,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())