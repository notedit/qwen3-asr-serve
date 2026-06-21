# qwen3-asr-serve

> 面向 **NVIDIA L20**（46 GB）的 **Qwen3-ASR-1.7B** + **Qwen3-ForcedAligner-0.6B** HTTP 生产服务，极致性能优先。

OpenAI 兼容的 `/v1/audio/transcriptions` + 自研 forced alignment 接口，含 batch 路由、文件路径直传（跳过 multipart 上传开销）、Prometheus 监控。

支持三种启动模式：**`asr`** / **`aligner`** / **`both`**（默认）。

## 核心特性

- **vLLM 0.14** 驱动 ASR。单卡 L20 实测 **~2,800 audio-s/s**（纯 ASR）/ **~480 audio-s/s**（带字级时间戳）。单条 p50 ~648 ms。
- **一键 Bash 安装**——无需 Docker。`./install.sh && ./run.sh`。
- **文件路径输入**——一等公民（`file_path=…`）。批量/大文件场景跳过 multipart 上传开销。
- **Aligner batch-pin** 内置（甜点值 16）——相比共享 batch 提升 +6.4% 吞吐、显存 -77%。
- **启动模式真隔离**：`MODE=asr` 时 aligner 不会加载，对齐路由返回 404；省下的显存可拉高 `gpu_memory_utilization`。
- **三个控制脚本**：`run.sh` / `stop.sh` / `status.sh`，支持后台 daemon 模式。

---

## 快速开始

```bash
# 1. 安装依赖 + 下载模型（约 6 GB）
./install.sh

# 2. 启动服务（默认 MODE=both）
./run.sh                      # 前台运行，ASR + Aligner 都启动
./run.sh asr                  # 前台，仅 ASR
./run.sh aligner              # 前台，仅 forced aligner

# Daemon 模式（后台运行，SSH 断开也存活，日志写入 ./logs/server.log）
./run.sh -d                   # 后台，MODE=both
./run.sh --daemon asr         # 后台，MODE=asr
./stop.sh                     # 优雅停止（SIGTERM，必要时升 SIGKILL）
./status.sh                   # 查看是否在跑，探测 /health，看日志尾部

# 3. 调用
curl -F file=@some.wav http://localhost:8000/v1/audio/transcriptions
```

默认监听 `0.0.0.0:8000`。通过 `.env`（复制 `.env.example` 修改）或环境变量配置。

### Daemon 模式细节

`./run.sh -d` 做了一个长期服务该做的事：

- 用 `setsid nohup` 启动，进程能在 shell 退出 / SSH 断开后存活
- 日志重定向到 `./logs/server.log`（stdout + stderr 合并）
- PID 写到 `./var/server.pid`，元数据（mode / port / 启动时间）写到 `./var/server.meta`
- 启动前端口冲突检查
- 上一个 daemon 还活着时拒绝再次启动——先 `./stop.sh`

`./stop.sh` 先 SIGTERM，等 30 秒（vLLM 释放 GPU 显存需要几秒），不行升 SIGKILL。
`./stop.sh --force` 跳过等待直接 SIGKILL。

如果用 supervisord / systemd 集成，**用前台模式**（`./run.sh`），由你的 supervisor 负责重启。

---

## API

所有路由有 OpenAPI 文档：`/docs`。挂载的路由依模式而定：

| 路由 | 方法 | `MODE=asr` | `MODE=aligner` | `MODE=both` |
| --- | --- | :---: | :---: | :---: |
| `/v1/audio/transcriptions` | POST | ✓ | — | ✓ |
| `/v1/audio/transcriptions/batch` | POST | ✓ | — | ✓ |
| `/v1/audio/forced_alignment` | POST | — | ✓ | ✓ |
| `/v1/audio/forced_alignment/batch` | POST | — | ✓ | ✓ |
| `/health`, `/metrics`, `/docs` | GET | ✓ | ✓ | ✓ |

### `POST /v1/audio/transcriptions`（OpenAI 兼容）

multipart 表单字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `file` | file | 与 `file_path` 二选一 |
| `file_path` | str | 本地路径；需开启 `ALLOW_LOCAL_PATHS=true` + `ALLOWED_PATH_PREFIXES` 白名单 |
| `model` | str | 接受但忽略（单租户） |
| `language` | str | ISO 639-1（`zh`、`en`、`yue`、`ja` ...）。不填即自动检测 |
| `response_format` | str | `json`（默认）/ `verbose_json` / `text` |
| `timestamp_granularities[]` | str | `word` 或 `segment` —— 都要就传两次；**需要 `MODE=both`** |

```bash
# 最简：multipart 上传
curl -F file=@a.wav http://localhost:8000/v1/audio/transcriptions

# 带字级时间戳，OpenAI verbose_json 形态
curl -F file=@a.wav \
     -F response_format=verbose_json \
     -F 'timestamp_granularities[]=word' \
     http://localhost:8000/v1/audio/transcriptions

# 性能版：传路径，跳过 multipart
curl -F file_path=/data/audio/a.wav \
     -F language=zh \
     http://localhost:8000/v1/audio/transcriptions
```

响应（verbose_json）：

```json
{
  "task": "transcribe",
  "language": "zh",
  "text": "刚收到快递，发现是男朋友…",
  "duration": 8.48,
  "words": [{"word":"刚","start":0.32,"end":0.48}, ...]
}
```

### `POST /v1/audio/transcriptions/batch`（扩展路由）

一次请求多文件，单次推理调用。传 `audio_files[]`（multipart）**或** `file_paths[]`（路径），共享 `language` / `response_format` / `timestamp_granularities[]`。

```bash
# multipart batch
curl -F 'audio_files=@a.wav' -F 'audio_files=@b.wav' \
     -F language=zh -F 'timestamp_granularities[]=word' \
     http://localhost:8000/v1/audio/transcriptions/batch

# 最佳吞吐：纯路径
curl -F 'file_paths=/data/audio/a.wav' \
     -F 'file_paths=/data/audio/b.wav' \
     -F language=zh \
     http://localhost:8000/v1/audio/transcriptions/batch
```

返回 `{"results": [...]}`，按上传顺序排列。

### `POST /v1/audio/forced_alignment`

单条 audio + 转写 → 字级时间戳。

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `file` 或 `file_path` | 是（二选一） | |
| `text` | 是 | 准确的转写文本 |
| `language` | 是 | ISO 码 |
| `granularity` | 否 | `word`（默认）/ `segment` |

```bash
curl -F file=@a.wav -F text="刚收到快递" -F language=zh \
     http://localhost:8000/v1/audio/forced_alignment
```

### `POST /v1/audio/forced_alignment/batch`

`audio_files[]` 或 `file_paths[]` + `texts[]`（与 audio 数量相同）+ `language` + `granularity`。

---

## 三种模式 —— 怎么选

| 模式 | 加载内容 | 适用场景 |
| --- | --- | --- |
| `asr` | 仅 Qwen3-ASR-1.7B（vLLM） | 纯转写。腾出来的显存可以拉高 `GPU_MEM_UTIL=0.85` 冲吞吐。 |
| `aligner` | 仅 Qwen3-ForcedAligner-0.6B（transformers） | 已有转写文本，只要时间戳。 |
| `both`（默认） | 都加载，aligner 与 ASR 共享 | 需要 ASR + 字级时间戳。 |

---

## 性能（NVIDIA L20，bf16）

数据源：上游 `perf_bench/`。工作负载：~8 秒中文 wav 复制凑批次，vLLM 后端，`gpu_memory_utilization=0.6`。

### 整体吞吐

| 模式 | 最佳 batch | `audio-s/s` | RTF | 日处理量（60% 利用率） |
| --- | ---: | ---: | ---: | ---: |
| 仅 ASR | 512 | **~2,800** | 0.000 | **~40,000 h/天** |
| ASR + 字级时间戳 | 256（aligner pin=16） | **~480** | 0.002 | **~6,800 h/天** |
| 仅 forced alignment | 16 | ~660 | 0.001 | ~9,500 h/天 |

### 单条延迟（batch=1）

| 操作 | p50 | p95 |
| --- | ---: | ---: |
| ASR | 648 ms | 688 ms |
| ASR + 时间戳 | 683 ms | 722 ms |

### Aligner-batch=16 为什么重要

直接让 aligner 跟着 `ASR_BATCH=256` 一起跑 → 486 audio-s/s，显存峰值 9.8 GB。
把 aligner 内部 batch 钉死到 16 → 517 audio-s/s，显存峰值 2.2 GB。
**同一份代码，吞吐 +6.4%，显存 -77%**，省出来的 7+ GB 可以给 vLLM 拉更大的 KV cache。

---

## 配置

所有环境变量（见 `.env.example`）：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MODE` | `both` | `asr` / `aligner` / `both` |
| `MODEL_DIR` | `./models` | 模型根目录 |
| `ASR_MODEL_PATH` | `./models/Qwen3-ASR-1.7B` | |
| `ALIGNER_MODEL_PATH` | `./models/Qwen3-ForcedAligner-0.6B` | |
| `GPU_MEM_UTIL` | `0.6` | `MODE=asr` 可调到 `0.85` |
| `ASR_BATCH` | `256` | vLLM Python 端分块大小 |
| `ALIGNER_BATCH` | `16` | aligner 内部 batch（实测甜点） |
| `HOST`, `PORT` | `0.0.0.0`, `8000` | |
| `ALLOW_LOCAL_PATHS` | `false` | 设为 `true` 才能用 `file_path` |
| `ALLOWED_PATH_PREFIXES` | 空 | 逗号分隔的绝对前缀；路径白名单 |
| `MAX_FILE_BYTES` | `209715200` | multipart 上传上限 200 MiB |
| `LOG_LEVEL` | `info` | |

---

## 监控（Metrics）

`GET /metrics` 返回 Prometheus exposition 格式。核心指标：

| 指标 | 标签 | 说明 |
| --- | --- | --- |
| `qwen_asr_requests_total` | `route`, `status`, `mode` | 各路由请求计数 |
| `qwen_asr_request_duration_seconds` | `route` | 直方图，PromQL `histogram_quantile` 求 p50/p95/p99 |
| `qwen_asr_inflight` | `route` | 在飞请求数 |
| `qwen_asr_audio_seconds_total` | `route` | 累计处理音频秒数 |
| `qwen_asr_batch_size` | `route` | 每次请求的 batch 大小直方图 |
| `qwen_asr_gpu_memory_bytes` | — | GPU 显存使用（每 5s 采样） |
| `qwen_asr_gpu_utilization_ratio` | — | GPU 利用率 [0,1]（每 5s 采样） |
| `qwen_asr_model_ready` | `mode`, `component` | 加载完成 = 1 |

PromQL 示例 —— transcriptions 路由的 p95 延迟：

```
histogram_quantile(0.95,
  sum(rate(qwen_asr_request_duration_seconds_bucket{route="transcriptions"}[5m]))
    by (le))
```

---

## 故障排查

| 现象 | 解决 |
| --- | --- |
| `torch.cuda.is_available() == False`，*Error 803* | 系统驱动版本和 `*/cuda*/compat/*` lib 不一致。`run.sh` 已自动剥离；若手动启 uvicorn 需自行剥掉 `LD_LIBRARY_PATH` 里含 `compat` 的条目。 |
| `pip install vllm==0.14.0` 报 "platform not supported" | 用官方 PyPI（`--index-url https://pypi.org/simple/`），加 `--platform manylinux_2_31_x86_64 --only-binary=:all:`。`install.sh` 已处理。 |
| `ImportError: cannot import name 'infer_schema' from 'torch.library'` | torch 版本 < 2.9，升级到 2.9.1（见 install.sh）。 |
| flash-attn `undefined symbol: at::_ops::zeros4call` | 系统的 flash-attn 2.6 与 torch 2.9 ABI 不兼容。`install.sh` 会卸载，vLLM 回退到内置 attention 实现。 |
| 模型下载超时 | `HF_ENDPOINT=https://hf-mirror.com python scripts/download_models.py --mode both` |
| 启动报 `Error retrieving safetensors … Network is unreachable` | vLLM 即便本地有模型也会 HEAD 探测 huggingface.co。`run.sh` 默认设了 `HF_HUB_OFFLINE=1`；若手动启动需自己设。 |
| `ValueError: ... KV cache memory ... larger than available` | 默认 `max_model_len=8192`（在 `lifespan.py`）应该够 ASR；若你拉高了请调小或调大 `GPU_MEM_UTIL`。 |
| `403 path is outside ALLOWED_PATH_PREFIXES` | 开启了 `ALLOW_LOCAL_PATHS` 但路径不在白名单内。把目录加进 `ALLOWED_PATH_PREFIXES`（逗号分隔的绝对路径）。 |

---

## API 测试

`scripts/test_api.sh` 是一个完整的 API 测试套件，覆盖所有路由、错误用例、边界条件：

```bash
BASE=http://127.0.0.1:8000 \
AUDIO_DIR=/path/to/audios \
bash scripts/test_api.sh
```

实测 35 项全过：4 个 ops 路由 + 8 个 ASR 正常用例 + 7 个 ASR 错误用例 + 6 个 batch 用例 + 3 个 alignment 正常 + 4 个 alignment 错误 + 3 个 alignment batch。

---

## 项目结构

```
.
├── install.sh                 vLLM 安装链 + 模型下载
├── run.sh                     启动 uvicorn（LD strip 内置；支持 -d daemon）
├── stop.sh                    优雅停止（SIGTERM + 必要时 SIGKILL）
├── status.sh                  状态查询 + /health 探测 + 日志尾部
├── pyproject.toml
├── .env.example
├── scripts/
│   ├── download_models.py     HF + hf-mirror fallback 下载
│   ├── verify.sh              curl 烟雾测试
│   └── test_api.sh            完整 API 测试套件
└── app/
    ├── main.py                FastAPI 工厂 + 按 MODE 挂路由
    ├── lifespan.py            加载 Qwen3ASRModel.LLM 和/或 Qwen3ForcedAligner
    ├── config.py              pydantic Settings（env 驱动）
    ├── aligner_patch.py       唯一验证过的优化（batch=16）
    ├── audio_io.py            multipart + 路径双输入
    ├── mapping.py             ASRTranscription ↔ OpenAI 形状，语言映射
    ├── metrics.py             Prometheus + GPU 采样
    ├── schemas.py             pydantic 响应模型
    └── routes/
        ├── transcriptions.py  /v1/audio/transcriptions [+ /batch]
        ├── alignments.py      /v1/audio/forced_alignment [+ /batch]
        └── ops.py             /health, /metrics
```

---

## License

Apache-2.0。封装上游 `qwen_asr` 包（同 Apache-2.0）。
