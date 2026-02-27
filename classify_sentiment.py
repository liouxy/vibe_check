#!/usr/bin/env python3
"""
评论情感分类工具
支持批量处理CSV中的评论，调用OpenAI API进行情感分类，并支持断点续传
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, Optional
import pandas as pd


try:
    from openai import OpenAI
except ImportError:
    print("请先安装依赖: pip install openai")
    exit(1)


def extract_json_from_llm(text):
    """
    不管 AI 是否带 ```json，都能准确把字典提取出来
    """
    # 1. 尝试匹配 Markdown 格式的 JSON 块
    # 匹配 ```json ... ``` 或者 ``` ... ``` 里的内容
    markdown_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(markdown_pattern, text)
    
    if match:
        json_str = match.group(1).strip()
    else:
        # 2. 如果没有 Markdown 标签，尝试寻找第一个 { 和最后一个 } 之间的内容
        # 这种方式可以过滤掉 AI 在 JSON 前后说的废话
        brace_pattern = r"(\{[\s\S]*\})"
        match = re.search(brace_pattern, text)
        if match:
            json_str = match.group(1).strip()
        else:
            # 3. 实在找不到，就死马当活马医，直接用原字符串
            json_str = text.strip()

    try:
        # 解析并返回 Python 字典
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"解析失败！原始文本: {text}")
        return None


class SentimentClassifier:
    """情感分类器"""
    
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        system_prompt_path: str = "prompts/nintendo_comment_classify.txt",
        cache_dir: str = "cache"
    ):
        """
        初始化分类器
        
        Args:
            api_key: OpenAI API密钥
            base_url: API端点URL，默认为None（使用官方endpoint）
            model: 使用的模型名称
            system_prompt_path: 系统提示词文件路径
            cache_dir: 缓存目录，用于断点续传
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.system_prompt = self._load_system_prompt(system_prompt_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def _load_system_prompt(self, prompt_path: str) -> str:
        """加载系统提示词"""
        prompt_file = Path(prompt_path)
        if not prompt_file.exists():
            raise FileNotFoundError(f"系统提示词文件不存在: {prompt_path}")
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def _get_cache_path(self, input_file: str) -> Path:
        """获取缓存文件路径"""
        input_name = Path(input_file).stem
        return self.cache_dir / f"{input_name}_cache.jsonl"
    
    def _load_cache(self, cache_path: Path) -> Dict[int, Dict]:
        """加载已处理的缓存"""
        cache = {}
        if cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    cache[item['index']] = item
        return cache
    
    def _save_to_cache(self, cache_path: Path, item: Dict):
        """追加保存到缓存文件"""
        with open(cache_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def classify_comment(self, comment: str) -> Dict:
        """
        对单条评论进行情感分类
        
        Args:
            comment: 评论文本
            
        Returns:
            分类结果字典
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": comment}
                ],
                temperature=0,
                max_tokens=1000
            )
            
            result = response.choices[0].message.content
            
            # 尝试解析为JSON
            parsed_result = extract_json_from_llm(result)
            
            return {
                "success": True,
                "result": parsed_result,
                "raw_response": result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": None
            }
    
    def process_csv(
        self,
        input_file: str,
        output_file: str,
        comment_field: str = "comment",
        max_retries: int = 3
    ):
        """
        处理CSV文件中的评论
        
        Args:
            input_file: 输入CSV文件路径
            output_file: 输出JSON文件路径
            comment_field: 评论字段名
            max_retries: 最大重试次数
        """
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_file}")
        
        # 加载缓存
        cache_path = self._get_cache_path(input_file)
        cache = self._load_cache(cache_path)
        print(f"已加载缓存: {len(cache)} 条记录")
        
        # 读取CSV
        with open(input_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        total = len(rows)
        print(f"总共 {total} 条评论需要处理")
        
        # 处理每条评论
        results = []
        for idx, row in enumerate(rows):
            # 检查是否已处理
            if idx in cache:
                print(f"[{idx+1}/{total}] 已缓存，跳过")
                results.append(cache[idx])
                continue
            
            comment = row.get(comment_field, "")
            if not comment.strip():
                print(f"[{idx+1}/{total}] 评论为空，跳过")
                continue
            
            # 重试机制
            success = False
            for retry in range(max_retries):
                print(f"[{idx+1}/{total}] 正在分类... (尝试 {retry+1}/{max_retries})")
                
                classification = self.classify_comment(comment)
                
                if classification["success"]:
                    item = {
                        "index": idx,
                        "comment": comment,
                        "raw_response": classification["raw_response"],
                        **{k: v for k, v in row.items() if k != comment_field}
                    }
                    item.update(classification["result"] or {})
                    # 保存到缓存
                    self._save_to_cache(cache_path, item)
                    results.append(item)
                    success = True
                    break
                else:
                    print(f"错误: {classification['error']}")
            
            if not success:
                print(f"[{idx+1}/{total}] 处理失败，跳过")
                item = {
                    "index": idx,
                    "comment": comment,
                    "raw_response": None,
                    "error": "error!",
                    **{k: v for k, v in row.items() if k != comment_field}
                }
                # self._save_to_cache(cache_path, item)
                results.append(item)
        
        # 保存最终结果
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # save csv with pandas

        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')


        
        print(f"\n处理完成! 结果已保存至: {output_file}")
        print(f"成功: {sum(1 for r in results if r.get('raw_response'))} 条")
        print(f"失败: {sum(1 for r in results if not r.get('raw_response'))} 条")


def main():
    parser = argparse.ArgumentParser(
        description="评论情感分类工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置处理单个文件
  python classify_sentiment.py -i inputs/comments.csv -o outputs/result.json
  
  # 指定自定义prompt和模型
  python classify_sentiment.py -i inputs/comments.csv -o outputs/result.json \\
    --prompt prompts/custom.txt --model gpt-4o
  
  # 使用自定义endpoint
  python classify_sentiment.py -i inputs/comments.csv -o outputs/result.json \\
    --base-url https://api.example.com/v1 --api-key YOUR_KEY
        """
    )
    
    # 基本参数
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='输入CSV文件路径'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='输出JSON文件路径'
    )
    parser.add_argument(
        '--comment-field',
        type=str,
        default='comment',
        help='CSV中评论字段名（默认: comment）'
    )
    
    # API配置
    parser.add_argument(
        '--api-key',
        type=str,
        default=os.getenv('OPENAI_API_KEY', ''),
        help='OpenAI API密钥（默认从环境变量OPENAI_API_KEY读取）'
    )
    parser.add_argument(
        '--base-url',
        type=str,
        default=None,
        help='API端点URL（默认使用官方endpoint）'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=os.getenv('DEFAULT_MODEL', 'gpt-4o-mini'),
        help='使用的模型（默认: gpt-4o-mini，支持环境变量 DEFAULT_MODEL）'
    )
    
    # Prompt配置
    parser.add_argument(
        '--prompt',
        type=str,
        default='prompts/nintendo_comment_classify.txt',
        help='系统提示词文件路径（默认: prompts/nintendo_comment_classify.txt）'
    )
    
    # 处理配置
    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='最大重试次数（默认: 3）'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='cache',
        help='缓存目录（默认: cache）'
    )
    
    args = parser.parse_args()
    print(args)
    
    # 检查API密钥
    if not args.api_key:
        print("错误: 请通过 --api-key 参数或 OPENAI_API_KEY 环境变量提供API密钥")
        return 1
    
    # 创建分类器
    try:
        classifier = SentimentClassifier(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            system_prompt_path=args.prompt,
            cache_dir=args.cache_dir
        )
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return 1
    
    # 处理文件
    classifier.process_csv(
        input_file=args.input,
        output_file=args.output,
        comment_field=args.comment_field,
        max_retries=args.max_retries
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
