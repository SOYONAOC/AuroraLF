# arXiv:2608.05531 复现工作区

目标论文：Ye-Hao Cheng, Yuan-Pei Yang, Ye Li, *A Roadmap for Transient
Hunters: Mapping Stellar Mass and Star Formation Rate Anisotropies in the
Local Universe*, arXiv:2608.05531v1。

当前状态（2026-08-08）：论文与 Zenodo 归档已核对；8 个公开文件均已下载并通过大小与 MD5 校验，3 个 ZIP 也已通过完整性测试并解包。独立 uv 环境已安装并锁定；可读版 ML notebook 已完成 Restart Kernel + Run All，天图代码尚未执行。ML 数值结果记录在 `data_save/reproductions/arxiv_2608_05531/ml/author_baseline/metrics.json`，不能据此标记整篇论文已复现。

完整的物理链条、Eqs. (1)–(7)、作者代码实际算法、图表验收门和公开缺口见 [`PHYSICS_AND_REPRODUCTION.md`](PHYSICS_AND_REPRODUCTION.md)。目前唯一已经独立执行的科学计算是 Eq. (5)：按论文打印参数得到 **1.545924113%**，而不是正文的 **1.78%**。

## 结论

作者整理数据起点的 ML 复跑难度为 **2/5**，不需要 GPU。完整公开下游流程（ML、HEALPix 天图和批量绘图）约为 **3/5**。从原始 REGALADE/GSWLC-2 目录端到端重建并严格复刻所有论文图表约为 **4–5/5**，而且当前公开材料仍缺少关键输入和分析代码。

| 目标 | 公开输入 | 建议 CPU / RAM | GPU | 工作磁盘 | 预计机器时间 | 预计人时 |
|---|---:|---:|---:|---:|---:|---:|
| ML 最小复跑 | 127.61 MiB | 8–16 核 / 8 GiB | 不需要 | 5 GiB | 2–10 分钟 | 2–6 小时 |
| ML 指标严格核对 | 165.03 MiB | 8–16 核 / 8–16 GiB | 不建议 | 5–10 GiB | 单次 2–10 分钟；版本矩阵 0.5–2 小时 | 8–24 小时 |
| 公开 ML + HEALPix notebooks | 248.85 MiB 全包 | 8–16 核 / 16 GiB | 不需要 | 5–10 GiB | 0.5–2 小时 | 2–4 天 |
| 补齐全论文 Fig. 1、5、6、7 | 部分输入与代码未归档 | 8–16 核 / 16 GiB | 不需要 | 10–20 GiB | 脚本完成后 1–3 小时 | 40–80 小时 |
| 7,988 万源目录端到端重建 | REGALADE 8.83 GB gzip | 24–32 核 / 64 GiB（分块） | 无益 | 150 GB 最低；300 GB 舒适 | 含下载与核验 1–3 天 | 40–120 小时 |

机器时间是工程估计。作者 notebook 保存的 CPU 运行记录为 124.47 秒，但论文没有报告作者机器的 CPU 型号、核数、RAM 或软件锁。REGALADE 官方主表含 79,880,104 条、每条 368 bytes，gzip 为 8,825,811,617 B，解压后含换行约 29.48 GB；150–300 GB 预算包括交叉匹配索引、临时表、环境和多版本输出。

## ML 复现对象

公开 notebook 的实际流程为：

1. 读取 `Model_training.csv` 的 335,264 行，将 object 列转为数值；非 object 列的所有 `0.0` 被替换为 NaN；仅按目标 `logSFR` 删除缺失行。目标中恰有 200 个零，因此剩余 335,064 行。论文没有说明这项零值规则；若零不是目录哨兵，`logSFR=0` 在物理上对应 SFR = 1 Msun/yr。
2. 得到 335,064 行，以 `random_seed=42` 随机切分为 223,376 行训练/验证集和 111,688 行 blind test。
3. 使用 7 个特征：`gmag`、`g-r`、`r-z`、`z-W1`、`W1-W2`、`redshift`、`logM`。
4. 在训练/验证集上做 shuffle 后的 10-fold OOF；每折均训练 CatBoost。
5. CatBoost 参数为 1200 棵树、深度 10、学习率 0.05、RMSE loss、随机种子 42、默认 CPU。
6. 只用 223,376 行训练/验证集再训练一次，评估 blind test，并预测 325,807 个缺少 SFR 的星系；111,688 行 blind 标签没有回灌，代码也没有在全部 335,064 个标签上重训生产模型。

本机最终运行复现了论文 Table 1 的显示精度以及 Zenodo notebook 的 R²/MAE。重跑的 325,807 个目录预测与发布值最大差 `4.44e-16 dex`，属于浮点解析精度。作者随机 split 是行级：52,384 个 blind 行（46.9%）的字符串 ObjID 也出现在训练侧，因此这里应称 row-held-out blind，不能解释为独立星系级盲测。

目标值和作者 notebook 输出已写入 `reproduction.toml`。论文 Table 1 对 blind test 正式报告 RMSE 0.306 dex、bias -0.0015 dex、$\sigma=0.306$ dex 和 3-sigma outlier fraction 2.302%；MAE 0.196 dex 与 $R^2=0.760$ 是 Zenodo notebook 的附加诊断，不在论文 Table 1 中。

## 目录边界

本文件夹只保存小型、可审计的控制文件。大文件遵守 AuroraLF 数据分层：

- 作者原始归档：`external_data/literature_sources/arxiv_2608_05531/zenodo_20695048/`
- 已解包作者产品：上述目录下的 `unpacked/{code,MASS,SFR}_archive/`
- 可复用复现结果：`data_save/reproductions/arxiv_2608_05531/`
- 日志与诊断图：`outputs/reproductions/arxiv_2608_05531/`
- 复现参数与预期指标：本目录的 `reproduction.toml`
- Zenodo 文件大小和 MD5：本目录的 `zenodo_manifest.json`
- 独立 Python 环境声明与锁：本目录的 `pyproject.toml`、`uv.lock`；本地 `.venv/` 不进入 Git

## 使用方式

在仓库根目录创建或同步本复现目录的独立环境：

```bash
uv sync --project scripts/experiments/arxiv_2608_05531 \
  --python 3.13.7 --no-install-project --no-build
```

只查看下载量：

```bash
PYTHONPATH=. scripts/experiments/arxiv_2608_05531/.venv/bin/python \
  scripts/experiments/arxiv_2608_05531/bootstrap.py --stage ml-verify
```

下载并逐文件核对大小与 Zenodo MD5。本机 uv Python 需要显式指向系统 CA：

```bash
SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt \
PYTHONPATH=. scripts/experiments/arxiv_2608_05531/.venv/bin/python \
  scripts/experiments/arxiv_2608_05531/bootstrap.py \
  --stage ml-verify --download
```

检查独立 `.venv` 是否具备 ML 依赖：

```bash
PYTHONPATH=. scripts/experiments/arxiv_2608_05531/.venv/bin/python \
  scripts/experiments/arxiv_2608_05531/audit_environment.py --stage ml
```

`requirements.in` 是根据作者 README/notebook 重建的兼容范围，不是作者发布的 lock file。`pyproject.toml` 和 `uv.lock` 记录本复现环境的实际声明与解析版本。

逐格阅读或运行 ML notebook：

```bash
cd scripts/experiments/arxiv_2608_05531
.venv/bin/jupyter lab ML_REPRODUCTION.ipynb
```

notebook 要求从这个小目录启动，避免猜测仓库路径。源 notebook 保持无输出，执行副本和 Fig. 2 放在 `outputs/reproductions/arxiv_2608_05531/ml/author_baseline/`；模型、逐行预测和指标放在 `data_save/reproductions/arxiv_2608_05531/ml/author_baseline/`。

## 当前环境与剩余阻塞项

本目录 `.venv` 使用 Python 3.13.7。核心解析版本为 NumPy 2.5.1、pandas 3.0.5、SciPy 1.18.0、scikit-learn 1.9.0、CatBoost 1.2.10、Matplotlib 3.11.1、healpy 1.20.0 和 Jupyter 1.1.1。依赖审计与 CatBoost 最小拟合均已通过。

公开归档占 260,933,673 B（248.85 MiB）。保留原文件并解包三个 ZIP 后，本机实测目录占 661,965,521 B（631.30 MiB，0.617 GiB）。5–10 GiB 建议值为环境、模型、缓存、复现表和诊断图保留了余量。

严格端到端复现仍缺：

- REGALADE 与 GSWLC-2 的原始目录快照、版本和校验值；
- TOPCAT 2 arcsec 匹配的一对多处理、join mode 和字段映射；
- 作者完整软件锁和 CPU 运行配置；
- Fig. 5 角功率谱、Fig. 6 相对涨落、Fig. 7 CCSN 对照的完整代码与目录快照；
- notebook 中写死的 `/Users/cyh/Mass_SFR/newVersion` 路径需要显式移植；
- 清洗步骤把全部数值零替换为 NaN，科学含义需在“原样对齐”完成后单独审计。

公开全包采用 MIT 许可证，Zenodo DOI：<https://doi.org/10.5281/zenodo.20695048>。论文：<https://arxiv.org/abs/2608.05531>。REGALADE 官方目录与记录规格：<https://cdsarc.cds.unistra.fr/ftp/J/A+A/706/A284/>、<https://cdsarc.cds.unistra.fr/ftp/J/A+A/706/A284/ReadMe>。GSWLC-2 官方目录：<https://cdsarc.cds.unistra.fr/ftp/J/ApJ/859/11/>。
