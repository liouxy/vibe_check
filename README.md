# Vibe Check - 评论情感分类工具

基于 OpenAI API 的评论情感分类工具，支持批量处理、断点续传和自定义提示词。

## 功能特性

- ✅ **批量处理**: 支持处理 CSV 文件中的大量评论
- ✅ **断点续传**: 自动缓存已处理结果，程序中断后可继续
- ✅ **自定义 Prompt**: 支持通过命令行指定系统提示词
- ✅ **灵活配置**: 支持自定义 API endpoint 和模型
- ✅ **重试机制**: 自动重试失败的请求
- ✅ **详细输出**: 保存原始 JSON 响应和结构化分类结果

## 目录结构

```
vibe_check/
├── classify_sentiment.py          # 主程序
├── requirements.txt                # 依赖列表
├── prompts/                        # 提示词目录
│   └── nintendo_comment_classify.txt  # 默认提示词
├── inputs/                         # 输入CSV文件目录
│   └── sample_comments.csv         # 示例数据
├── outputs/                        # 输出JSON文件目录
└── cache/                          # 缓存目录（断点续传）
```

## 安装

### 1. 克隆仓库

```bash
git clone <repository-url>
cd vibe_check
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API 密钥

```bash
# 方法1: 设置环境变量（推荐）
export OPENAI_API_KEY="your-api-key-here"

# 方法2: 或在运行时通过命令行参数指定
```

## 使用方法

### 基本用法

```bash
# 处理CSV文件
python classify_sentiment.py \
  -i inputs/comments.csv \
  -o outputs/result.json
```

### 自定义配置

```bash
# 使用自定义提示词和模型
python classify_sentiment.py \
  -i inputs/comments.csv \
  -o outputs/result.json \
  --prompt prompts/custom_prompt.txt \
  --model gpt-4o

# 使用自定义 API endpoint
python classify_sentiment.py \
  -i inputs/comments.csv \
  -o outputs/result.json \
  --base-url https://api.example.com/v1 \
  --api-key YOUR_API_KEY
```

### 完整参数说明

```bash
python classify_sentiment.py --help
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-i, --input` | 输入CSV文件路径 | 必需 |
| `-o, --output` | 输出JSON文件路径 | 必需 |
| `--comment-field` | CSV中评论字段名 | `comment` |
| `--api-key` | OpenAI API密钥 | 从环境变量读取 |
| `--base-url` | API endpoint URL | 官方endpoint |
| `--model` | 使用的模型 | `gpt-4o-mini` |
| `--prompt` | 系统提示词文件路径 | `prompts/nintendo_comment_classify.txt` |
| `--delay` | 请求间隔（秒） | `0.5` |
| `--max-retries` | 最大重试次数 | `3` |
| `--cache-dir` | 缓存目录 | `cache` |

## 输入格式

CSV 文件需要包含评论字段（默认为 `comment`），示例：

```csv
id,comment,user,date
1,塞尔达传说王国之泪太好玩了！画面精美，玩法创新，强烈推荐！,user_001,2024-05-15
2,这代宝可梦优化太差了，帧数掉得厉害，体验很糟糕,user_002,2024-03-20
```

## 输出格式

输出为 JSON 文件，包含每条评论的分类结果：

```json
[
  {
    "index": 0,
    "comment": "塞尔达传说王国之泪太好玩了...",
    "classification": {
      "sentiment": "positive",
      "confidence": 0.95,
      "aspects": {
        "gameplay": "positive",
        "graphics": "positive",
        "story": "not_mentioned",
        "music": "not_mentioned",
        "value": "positive"
      },
      "keywords": ["好玩", "画面精美", "创新", "推荐"],
      "summary": "用户对游戏整体评价非常积极"
    },
    "raw_response": "{...}",
    "id": "1",
    "user": "user_001",
    "date": "2024-05-15"
  }
]
```

## 断点续传机制

程序会在 `cache/` 目录下创建缓存文件（`.jsonl` 格式），每处理一条评论就立即保存。如果程序中断：

1. 重新运行相同的命令
2. 程序会自动加载缓存
3. 跳过已处理的评论
4. 从中断处继续

缓存文件示例：`cache/comments_cache.jsonl`

## 自定义 Prompt

创建自己的提示词文件（例如 `prompts/my_prompt.txt`）：

```
你是一个情感分析专家...

请按照以下JSON格式返回结果：
{
  "sentiment": "...",
  "keywords": [...]
}
```

然后使用：

```bash
python classify_sentiment.py \
  -i inputs/comments.csv \
  -o outputs/result.json \
  --prompt prompts/my_prompt.txt
```

## 注意事项

1. **API 配额**: 大批量处理请注意 API 调用限制
2. **成本控制**: 建议先用小数据集测试
3. **网络稳定性**: 如果网络不稳定，可增加 `--max-retries` 和 `--delay`
4. **缓存管理**: 如需重新处理，删除对应的缓存文件

## 示例工作流

```bash
# 1. 准备数据
# 将爬取的评论CSV放入 inputs/ 目录

# 2. 设置API密钥
export OPENAI_API_KEY="sk-..."

# 3. 处理评论
python classify_sentiment.py \
  -i inputs/comments.csv \
  -o outputs/result.json

# 4. 如果程序中断，重新运行相同命令即可从断点继续
```

# 4. 查看结果
# 结果保存在 outputs/ 目录下
```

## 故障排查

### 问题：API 密钥错误
```bash
# 检查环境变量
echo $OPENAI_API_KEY

# 或直接指定
python classify_sentiment.py --api-key "sk-..."
```

### 问题：找不到提示词文件
```bash
# 确认文件存在
ls prompts/nintendo_comment_classify.txt

# 或使用绝对路径
--prompt /absolute/path/to/prompt.txt
```

### 问题：CSV 字段名不匹配
```bash
# 指定正确的字段名
--comment-field "评论内容"
```

## License

MIT