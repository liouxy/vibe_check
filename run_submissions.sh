#!/bin/bash

# 快速运行脚本示例

# 加载环境变量（如果使用 .env 文件）
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi


# 或者指定单个文件
/usr/local/bin/python3 classify_sentiment.py \
  -i inputs/nintendo_submissions_sample30.jsonl \
  -o outputs/nintendo_submissions_sample30.csv \
  --comment-fields title,selftext \
  --model gpt-5-mini
