# 沪深300深度学习量化实验

本目录包含《深度学习基础大作业》的可复现实验代码。当前实现仅使用沪深300股票池，模型输入为历史日频特征序列，输出横截面股票分数，并通过等权组合进行模拟回测。

## 环境

使用 `ai25` Conda 环境：

```powershell
conda run -n ai25 python -m compileall code
```

主要依赖见 `requirements.txt`：`numpy`、`pandas`、`torch`、`tqdm`、`openpyxl`、`matplotlib`。

## 数据

当前实验默认复用：

```text
data/panel_hs300_advanced.npz
```

该面板由 `A股数据/daily`、`A股数据/metric`、`A股数据/moneyflow` 和 `code/hs300_constituents.csv` 生成，包含：

- `features`: `(交易日, 股票数, 特征数)`
- `returns`: 未来 1 日 open-to-open 收益，即特征日 `t` 盘后打分，`t+1` 开盘买入，`t+2` 开盘卖出
- `masks`: 收益是否有效
- `dates`、`stocks`、`feature_cols`

如需重新生成面板：

```powershell
conda run -n ai25 python code/load_data.py `
  --data_dir "A股数据/daily" `
  --metric_dir "A股数据/metric" `
  --moneyflow_dir "A股数据/moneyflow" `
  --pool hs300 `
  --stock_pool_file code/hs300_constituents.csv `
  --top_k 300 `
  --out data/panel_hs300_advanced.npz
```

## 一键实验

正式实验入口：

```powershell
conda run -n ai25 python code/run_experiments.py `
  --refresh_panel never `
  --epochs 20 `
  --batch_size 64 `
  --seq_len 30 `
  --top_n 10 `
  --k 3 `
  --fee 0.0005 `
  --device cuda
```

输出目录形如：

```text
experiments/hs300_YYYYMMDD_HHMMSS/
```

其中包含：

- `experiment_report.md`: 中文实验报告
- `summary_metrics.csv`: 各实验核心指标
- `interval_returns_10d.csv`: 2026 年 3 月以来每 10 个交易日分段统计
- `nav_compare.png`、`improved_nav_compare.png`: NAV 曲线
- `latest_weights_transformer.csv`: 最新日期的 Top 10 等权持仓建议
- 各模型子目录中的 `best.pt`、`history.csv`、`loss_ic_curve.png`、回测 CSV、持仓 CSV、metrics JSON

## 模型与策略

`model.py` 支持：

- `transformer`
- `gru`
- `lstm`
- `transformer_cs`

基线策略统一为：

- 初始建仓：预测分数 Top 10 等权买入
- 后续每日：Sell 3 / Buy 3
- 始终等权满仓
- 默认单边手续费 `0.0005`

改进实验使用：

- 5 日 forward open-to-open 复合收益标签
- 趋势过滤
- 波动惩罚
- 资金流修正
- 止损替换
- 5 日主调仓

## 单独训练

```powershell
conda run -n ai25 python code/train.py `
  --panel data/panel_hs300_advanced.npz `
  --model transformer `
  --seq_len 30 `
  --label_horizon 1 `
  --train_start 2016-01-04 `
  --train_end 2025-12-31 `
  --test_start 2026-01-01 `
  --test_end 2026-02-27 `
  --epochs 20 `
  --batch_size 64 `
  --top_n 10 `
  --out_dir runs/transformer_hs300
```

## 单独回测

```powershell
conda run -n ai25 python code/back_test_equal.py `
  --panel data/panel_hs300_advanced.npz `
  --checkpoint runs/transformer_hs300/best.pt `
  --seq_len 30 `
  --start 2026-03-02 `
  --end 2026-05-27 `
  --top_n 10 `
  --k 3 `
  --fee 0.0005 `
  --out backtest.csv
```

最近 10 个可完整回测决策日为 `2026-05-14` 至 `2026-05-27`。面板最新日 `2026-05-29` 可用于生成下一交易日预测持仓，但尚不能计算完整真实收益。

## 最新持仓预测

```powershell
conda run -n ai25 python code/predict_weights.py `
  --panel data/panel_hs300_advanced.npz `
  --checkpoint experiments/hs300_YYYYMMDD_HHMMSS/transformer/best.pt `
  --seq_len 30 `
  --top_n 10 `
  --equal_weight `
  --out latest_weights_transformer.csv
```

## 防止数据泄露

- 特征只使用决策日 `t` 及之前的数据。
- 交易信号在 `t` 日盘后生成，执行发生在 `t+1` 日开盘。
- 标签使用未来 open-to-open 收益，不进入特征标准化。
- 数据集按日期切分，不随机混合训练和验证日期。
- 横截面标准化按单个交易日计算，不使用未来日期统计量。
