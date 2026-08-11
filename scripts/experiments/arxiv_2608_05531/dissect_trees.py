"""拆解 catboost_final.cbm 的 1200 棵树（教学用诊断脚本）。

回答三个问题：
1. 模型内部到底存了什么？(bias + 1200 棵对称树)
2. 第一棵树的 10 条分裂规则是什么？怎么选出来的？
3. 1200 棵树如何接力收敛？(累计 RMSE 曲线 + 单星系逐树追踪)

运行方式（从本目录）：
    .venv/bin/python dissect_trees.py
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA_DIR = REPO / "external_data/literature_sources/arxiv_2608_05531/zenodo_20695048"
RESULT_DIR = REPO / "data_save/reproductions/arxiv_2608_05531/ml/author_baseline"
MODEL_PATH = RESULT_DIR / "catboost_final.cbm"

FEATURES = ["g-r", "r-z", "z-W1", "W1-W2", "redshift", "gmag", "logM"]
TARGET = "logSFR"

# ---------------------------------------------------------------------------
# 1. 读模型 + 训练数据（与 notebook 完全相同的清洗和切分）
# ---------------------------------------------------------------------------
model = CatBoostRegressor()
model.load_model(str(MODEL_PATH))
scale, bias = model.get_scale_and_bias()

training = pd.read_csv(DATA_DIR / "Model_training.csv", dtype={"ObjID": "string"})
numeric_columns = training.select_dtypes(include="number").columns
training[numeric_columns] = training[numeric_columns].replace(0.0, np.nan)
training = training.replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET])
# notebook cell 8: 颜色 = 星等差
training["g-r"] = training["gmag"] - training["rmag"]
training["r-z"] = training["rmag"] - training["zmag"]
training["z-W1"] = training["zmag"] - training["W1mag"]
training["W1-W2"] = training["W1mag"] - training["W2mag"]
train_val, blind = train_test_split(
    training, test_size=1 / 3, random_state=42, shuffle=True
)
Xtv = train_val[FEATURES].to_numpy()
ytv = train_val[TARGET].to_numpy()
Xbl = blind[FEATURES].to_numpy()
ybl = blind[TARGET].to_numpy()

# 模型内部结构：导出为 JSON
tmp_json = Path("/tmp/catboost_model.json")
model.save_model(str(tmp_json), format="json")
with open(tmp_json) as f:
    doc = json.load(f)
trees = doc["oblivious_trees"]
assert len(trees) == 1200

# 特征名映射（优先用模型自己的记录，退回 FEATURES 顺序）
try:
    fnames = {
        f["feature_index"]: f.get("feature_name", FEATURES[f["feature_index"]])
        for f in doc["features_info"]["float_features"]
    }
except (KeyError, IndexError):
    fnames = {i: name for i, name in enumerate(FEATURES)}

print("=" * 78)
print("第 0 步：模型内部存了什么")
print("=" * 78)
print(f"scale = {scale},  bias(起点) = {bias:.6f}")
print(f"树的总数 = {len(trees)}")
print(f"每棵树 = {len(trees[0]['splits'])} 条分裂规则 x "
      f"{len(trees[0]['leaf_values'])} 个叶子值 (2^10)")
print("预测公式 = bias + 树1的叶值 + 树2的叶值 + ... + 树1200的叶值\n")

# ---------------------------------------------------------------------------
# 2. 校验：叶子值确实就是"预测增量"，且 bias 就是起点
# ---------------------------------------------------------------------------
print("=" * 78)
print("第 0.5 步：用数据验证上面的公式（只取无缺失特征的行）")
print("=" * 78)
good = ~np.isnan(Xtv).any(axis=1)
Xg, yg = Xtv[good], ytv[good]
stages = np.array(list(model.staged_predict(Xg)))  # (1200, n): 逐树累计预测
for t in range(3):
    delta = stages[t] - (stages[t - 1] if t else bias)
    leaf_v = np.asarray(trees[t]["leaf_values"], dtype=float)[
        np.asarray(model.calc_leaf_indexes(Xg, t))[:, 0]
    ]
    print(f"  树 {t + 1:3d}: max|叶值 - 预测增量| = {np.abs(leaf_v - delta).max():.2e}")
print(f"  bias + 1200 棵树之和 与 predict 之差: "
      f"max|stages[-1] - predict| = {np.abs(stages[-1] - model.predict(Xg)).max():.2e}")
print()

# ---------------------------------------------------------------------------
# 3. 第一棵树的 10 条分裂规则
# ---------------------------------------------------------------------------
print("=" * 78)
print("第 1 步：第一棵树的 10 条分裂规则（从上到下，整层共用一条）")
print("=" * 78)
tree0 = trees[0]
for d, sp in enumerate(tree0["splits"]):
    f = sp["float_feature_index"]
    print(f"  第{d + 1:2d}层  {fnames[f]:<10s} > {sp['border']:.4f}  则向右")
print()

# 第一层的物理含义：均值模型在 W1-W2 两侧各高估/低估多少
r1 = ytv - bias
side_left = Xtv[:, 3] <= tree0["splits"][0]["border"]
print("  第一层分裂的物理含义（目标 = 残差 r1 = logSFR - 起点）:")
sp0 = tree0["splits"][0]
print(f"    W1-W2 <= {sp0['border']:.4f} 一侧: "
      f"n={side_left.sum()}, mean(r1)={r1[side_left].mean():+.4f} dex")
print(f"    W1-W2 >  {sp0['border']:.4f} 一侧: "
      f"n={(~side_left).sum()}, mean(r1)={r1[~side_left].mean():+.4f} dex")
print()

# 叶值分布
vals0 = np.asarray(tree0["leaf_values"], dtype=float)
print("  第一棵树的 1024 个叶值:")
print(f"    min={vals0.min():+.4f}  max={vals0.max():+.4f}  "
      f"median={np.median(vals0):+.4f}")
print(f"    严格为零的叶子: {(vals0 == 0).sum()}  |v|>0.005 的叶子: "
      f"{(np.abs(vals0) > 0.005).sum()}")
print()

# ---------------------------------------------------------------------------
# 4. 验证"第一个分裂是怎么选出来的"：暴力搜索最小化残差平方和
# ---------------------------------------------------------------------------
def best_split(X, r):
    """对目标 r 做暴力搜索：在全部特征、全部候选切割点上找
    使 总SSE - 左SSE - 右SSE 最大的那个 (特征, 边界)。"""
    total_sse = np.sum(r**2) - np.sum(r) ** 2 / len(r)
    best = None
    for f in range(X.shape[1]):
        x = X[:, f]
        m = ~np.isnan(x)
        if m.sum() < 20:
            continue
        xs, rs = x[m], r[m]
        order = np.argsort(xs, kind="stable")
        xs, rs = xs[order], rs[order]
        cs, css = np.cumsum(rs), np.cumsum(rs**2)
        n = len(xs)
        # 候选边界：相邻两个数据点的中点
        pos = np.arange(1, n)
        nL, nR = pos, n - pos
        sL, s2L = cs[pos - 1], css[pos - 1]
        sR, s2R = cs[-1] - sL, css[-1] - s2L
        sseL = s2L - sL**2 / nL
        sseR = s2R - sR**2 / nR
        gain = total_sse - sseL - sseR
        j = int(np.argmax(gain))
        border = 0.5 * (xs[j] + xs[j + 1])
        if best is None or gain[j] > best[0]:
            best = (gain[j], f, border)
    return best


print("=" * 78)
print("第 2 步：第一层分裂是暴力搜出来的吗？(最小化 SSE = 残差平方和)")
print("=" * 78)
gain, f_best, border_best = best_split(Xtv, r1)
print(f"  暴力搜索最优: {fnames[f_best]:<10s} 边界 {border_best:.4f}, "
      f"SSE 减少 {gain:.0f}")
sp0 = tree0["splits"][0]
print(f"  模型实际使用: {fnames[sp0['float_feature_index']]:<10s} 边界 "
      f"{sp0['border']:.4f}")
match = (
    fnames[f_best] == fnames[sp0["float_feature_index"]]
    and abs(border_best - sp0["border"]) < 0.05
)
print(f"  一致？ {match}")
print("  (说明: CatBoost 在预先算好的量化网格上搜，所以边界会有细微差别)")
print()

# ---------------------------------------------------------------------------
# 5. 第二棵树在修什么
# ---------------------------------------------------------------------------
print("=" * 78)
print("第 3 步：第二棵树的起点变了——它修的是残差 r2 = y - (bias + 树1)")
print("=" * 78)
r2 = ytv - model.predict(Xtv, ntree_end=1)
gain2, f2, b2 = best_split(Xtv, r2)
sp1 = trees[1]["splits"][0]
print(f"  暴力搜索 r2 的最优分裂: {fnames[f2]:<10s} 边界 {b2:.4f}")
print(f"  树2 实际的第一层分裂:   {fnames[sp1['float_feature_index']]:<10s} "
      f"边界 {sp1['border']:.4f}")
print(f"  树1 第一层用的是        {fnames[tree0['splits'][0]['float_feature_index']]:<10s}")
print()

# ---------------------------------------------------------------------------
# 6. 累计学习曲线：前 k 棵树预测的 RMSE（train-val 和 blind 并排）
# ---------------------------------------------------------------------------
print("=" * 78)
print("第 4 步：用前 k 棵树预测，RMSE 怎么变 (k=0 表示只输出起点)")
print("=" * 78)
ks = [0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000, 1200]
print(f"  {'k':>5s}  {'RMSE(train-val)':>16s}  {'RMSE(blind)':>13s}")
for k in ks:
    p_tv = model.predict(Xtv, ntree_end=k) if k else np.full(len(ytv), bias)
    p_bl = model.predict(Xbl, ntree_end=k) if k else np.full(len(ybl), bias)
    r_tv = np.sqrt(np.mean((p_tv - ytv) ** 2))
    r_bl = np.sqrt(np.mean((p_bl - ybl) ** 2))
    print(f"  {k:>5d}  {r_tv:>16.4f}  {r_bl:>13.4f}")
print()

# ---------------------------------------------------------------------------
# 7. 挑一个具体星系，逐树追踪它的预测
# ---------------------------------------------------------------------------
print("=" * 78)
print("第 5 步：跟住一个星系，看 1200 棵树怎么把它推到最终预测")
print("=" * 78)
good_bl = ~np.isnan(Xbl).any(axis=1)
row = int(np.where(good_bl)[0][0])
x0 = Xbl[row : row + 1]
y0 = float(ybl[row])
print(f"  目标星系 (blind 第 {row} 行):")
print(f"    特征: " + ", ".join(f"{FEATURES[i]}={x0[0, i]:.3f}" for i in range(7)))
print(f"    GSWLC-2 真值 logSFR = {y0:.3f}")
print(f"    起点 bias = {bias:.4f}")
cum = bias
print(f"  {'t':>5s} {'累计预测':>10s} {'真值-预测':>10s}")
print(f"  {0:>5d} {bias:>10.4f} {y0 - bias:>+10.4f}")
bl_stages = np.array(list(model.staged_predict(x0)))
for t in range(12):
    cum = bl_stages[t, 0]
    print(f"  {t + 1:>5d} {cum:>10.4f} {y0 - cum:>+10.4f}")
for t in [99, 499, 1199]:
    p = bl_stages[t, 0]
    print(f"  {t + 1:>5d} {p:>10.4f} {y0 - p:>+10.4f}")
final = bl_stages[-1, 0]
print(f"  最终预测 = {final:.4f}  真值 = {y0:.4f}")
