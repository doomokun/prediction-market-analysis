"""
BTC Up/Down 5m Market Probability Analysis

純統計、概率方式分析 btc-updown-5m-* markets。
分步驟逐層分析：
1. 基礎統計事實
2. 時間維度分析
3. 連續性/動量分析
4. 橫向對比
5. 統計顯著性驗證
6. 總結洞見
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class StatResult:
    """統計結果容器"""
    name: str
    value: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    p_value: float | None = None
    n: int = 0
    significant: bool | None = None

    def __repr__(self) -> str:
        ci_str = f" [{self.ci_lower:.4f}, {self.ci_upper:.4f}]" if self.ci_lower else ""
        p_str = f" (p={self.p_value:.4f})" if self.p_value else ""
        sig_str = " *" if self.significant else ""
        return f"{self.name}: {self.value:.4f}{ci_str}{p_str} n={self.n}{sig_str}"


@dataclass
class AnalysisStep:
    """分析步驟結果"""
    step_num: int
    title: str
    findings: list[str] = field(default_factory=list)
    stats: list[StatResult] = field(default_factory=list)
    data: pd.DataFrame | None = None

    def summary(self) -> str:
        lines = [f"\n{'='*60}", f"Step {self.step_num}: {self.title}", "="*60]
        for stat in self.stats:
            lines.append(f"  {stat}")
        for finding in self.findings:
            lines.append(f"  → {finding}")
        return "\n".join(lines)


class BTCUpDownProbabilityAnalysis:
    """BTC Up/Down 概率統計分析"""

    def __init__(self, markets_dir: Path | str | None = None):
        base_dir = Path(__file__).parent.parent.parent.parent
        self.markets_dir = Path(markets_dir or base_dir / "data" / "polymarket" / "markets")
        self.df: pd.DataFrame | None = None
        self.steps: list[AnalysisStep] = []

    def load_data(self) -> pd.DataFrame:
        """載入並預處理數據"""
        con = duckdb.connect()

        df = con.execute(f"""
            SELECT
                slug,
                question,
                outcomes,
                outcome_prices,
                end_date,
                volume,
                liquidity,
                created_at
            FROM '{self.markets_dir}/*.parquet'
            WHERE lower(slug) LIKE 'btc-updown-5m%'
            ORDER BY end_date
        """).df()

        # Parse resolution
        def parse_winner(outcome_prices: str) -> str | None:
            try:
                import json
                prices = json.loads(outcome_prices)
                if len(prices) != 2:
                    return None
                p0, p1 = float(prices[0]), float(prices[1])
                # Need clear resolution
                if max(p0, p1) < 0.99:
                    return None
                return "Up" if p0 > p1 else "Down"
            except:
                return None

        df["winner"] = df["outcome_prices"].apply(parse_winner)
        df = df[df["winner"].notna()].copy()

        # Extract timestamp from slug (btc-updown-5m-TIMESTAMP)
        df["timestamp"] = df["slug"].str.extract(r"btc-updown-5m-(\d+)")[0].astype(float)
        df["end_dt"] = pd.to_datetime(df["end_date"], utc=True)
        df["hour"] = df["end_dt"].dt.hour
        df["dow"] = df["end_dt"].dt.dayofweek  # 0=Mon, 6=Sun
        df["dow_name"] = df["end_dt"].dt.day_name()
        df["date"] = df["end_dt"].dt.date
        df["is_up"] = (df["winner"] == "Up").astype(int)

        # Sort by time
        df = df.sort_values("timestamp").reset_index(drop=True)

        self.df = df
        return df

    def step1_basic_stats(self) -> AnalysisStep:
        """Step 1: 基礎統計事實"""
        step = AnalysisStep(1, "基礎統計事實 (Baseline Statistics)")
        df = self.df

        n_total = len(df)
        n_up = df["is_up"].sum()
        n_down = n_total - n_up

        # Overall win rate
        up_rate = n_up / n_total

        # Binomial test vs 50%
        binom_result = stats.binomtest(n_up, n_total, p=0.5)
        ci = binom_result.proportion_ci(confidence_level=0.95)

        step.stats.append(StatResult(
            name="總市場數",
            value=n_total,
            n=n_total
        ))

        step.stats.append(StatResult(
            name="Up 勝出數",
            value=n_up,
            n=n_up
        ))

        step.stats.append(StatResult(
            name="Down 勝出數",
            value=n_down,
            n=n_down
        ))

        step.stats.append(StatResult(
            name="Up 勝率 (vs 50%)",
            value=up_rate,
            ci_lower=ci.low,
            ci_upper=ci.high,
            p_value=binom_result.pvalue,
            n=n_total,
            significant=binom_result.pvalue < 0.05
        ))

        # Findings
        if binom_result.pvalue < 0.05:
            direction = "高於" if up_rate > 0.5 else "低於"
            step.findings.append(f"Up 勝率顯著{direction} 50% (p={binom_result.pvalue:.4f})")
        else:
            step.findings.append(f"Up 勝率與 50% 無顯著差異 (p={binom_result.pvalue:.4f})")

        # Daily stats
        daily_up_rate = df.groupby("date")["is_up"].agg(["sum", "count", "mean"])
        step.stats.append(StatResult(
            name="日均市場數",
            value=daily_up_rate["count"].mean(),
            n=len(daily_up_rate)
        ))

        step.stats.append(StatResult(
            name="Up 勝率標準差 (日)",
            value=daily_up_rate["mean"].std(),
            n=len(daily_up_rate)
        ))

        self.steps.append(step)
        return step

    def step2_time_analysis(self) -> AnalysisStep:
        """Step 2: 時間維度分析"""
        step = AnalysisStep(2, "時間維度分析 (Temporal Patterns)")
        df = self.df

        # Hourly analysis
        hourly = df.groupby("hour")["is_up"].agg(["sum", "count", "mean"])
        hourly.columns = ["up_count", "total", "up_rate"]

        # Chi-square test for hourly independence
        chi2, p_hour = stats.chisquare(hourly["up_count"],
                                        f_exp=hourly["total"] * df["is_up"].mean())

        step.stats.append(StatResult(
            name="小時獨立性 Chi² 檢驗",
            value=chi2,
            p_value=p_hour,
            n=24,
            significant=p_hour < 0.05
        ))

        # Find best/worst hours
        best_hour = hourly["up_rate"].idxmax()
        worst_hour = hourly["up_rate"].idxmin()

        step.stats.append(StatResult(
            name=f"最高 Up 率小時 ({best_hour}:00)",
            value=hourly.loc[best_hour, "up_rate"],
            n=int(hourly.loc[best_hour, "total"])
        ))

        step.stats.append(StatResult(
            name=f"最低 Up 率小時 ({worst_hour}:00)",
            value=hourly.loc[worst_hour, "up_rate"],
            n=int(hourly.loc[worst_hour, "total"])
        ))

        # Day of week analysis
        dow_stats = df.groupby("dow")["is_up"].agg(["sum", "count", "mean"])
        dow_stats.columns = ["up_count", "total", "up_rate"]
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        chi2_dow, p_dow = stats.chisquare(dow_stats["up_count"],
                                           f_exp=dow_stats["total"] * df["is_up"].mean())

        step.stats.append(StatResult(
            name="星期獨立性 Chi² 檢驗",
            value=chi2_dow,
            p_value=p_dow,
            n=7,
            significant=p_dow < 0.05
        ))

        best_dow = dow_stats["up_rate"].idxmax()
        worst_dow = dow_stats["up_rate"].idxmin()

        step.stats.append(StatResult(
            name=f"最高 Up 率星期 ({dow_names[best_dow]})",
            value=dow_stats.loc[best_dow, "up_rate"],
            n=int(dow_stats.loc[best_dow, "total"])
        ))

        step.stats.append(StatResult(
            name=f"最低 Up 率星期 ({dow_names[worst_dow]})",
            value=dow_stats.loc[worst_dow, "up_rate"],
            n=int(dow_stats.loc[worst_dow, "total"])
        ))

        # Findings
        if p_hour < 0.05:
            step.findings.append(f"小時存在顯著模式 - {best_hour}:00 Up率最高 ({hourly.loc[best_hour, 'up_rate']:.1%})")
        else:
            step.findings.append("小時維度無顯著模式")

        if p_dow < 0.05:
            step.findings.append(f"星期存在顯著模式 - {dow_names[best_dow]} Up率最高 ({dow_stats.loc[best_dow, 'up_rate']:.1%})")
        else:
            step.findings.append("星期維度無顯著模式")

        step.data = hourly
        self.steps.append(step)
        return step

    def step3_momentum_analysis(self) -> AnalysisStep:
        """Step 3: 連續性/動量分析"""
        step = AnalysisStep(3, "連續性/動量分析 (Momentum & Streaks)")
        df = self.df.copy()

        # Previous N outcomes
        for lag in range(1, 11):
            df[f"prev_{lag}"] = df["is_up"].shift(lag)

        # After Up, probability of Up
        mask_after_up = df["prev_1"] == 1
        after_up_rate = df.loc[mask_after_up, "is_up"].mean()
        after_up_n = mask_after_up.sum()

        # After Down, probability of Up
        mask_after_down = df["prev_1"] == 0
        after_down_rate = df.loc[mask_after_down, "is_up"].mean()
        after_down_n = mask_after_down.sum()

        step.stats.append(StatResult(
            name="P(Up | 上次 Up)",
            value=after_up_rate,
            n=int(after_up_n)
        ))

        step.stats.append(StatResult(
            name="P(Up | 上次 Down)",
            value=after_down_rate,
            n=int(after_down_n)
        ))

        # Test if significantly different
        chi2_momentum, p_momentum = stats.chi2_contingency([
            [df.loc[mask_after_up, "is_up"].sum(), (mask_after_up & (df["is_up"]==0)).sum()],
            [df.loc[mask_after_down, "is_up"].sum(), (mask_after_down & (df["is_up"]==0)).sum()]
        ])[:2]

        step.stats.append(StatResult(
            name="一階馬可夫獨立性檢驗",
            value=chi2_momentum,
            p_value=p_momentum,
            significant=p_momentum < 0.05
        ))

        # Streak analysis
        df["streak_start"] = (df["is_up"] != df["is_up"].shift(1)).astype(int)
        df["streak_id"] = df["streak_start"].cumsum()
        streaks = df.groupby("streak_id").agg({
            "is_up": ["first", "count"]
        })
        streaks.columns = ["direction", "length"]

        up_streaks = streaks[streaks["direction"] == 1]["length"]
        down_streaks = streaks[streaks["direction"] == 0]["length"]

        step.stats.append(StatResult(
            name="Up 連勝平均長度",
            value=up_streaks.mean(),
            n=len(up_streaks)
        ))

        step.stats.append(StatResult(
            name="Up 連勝最長",
            value=up_streaks.max(),
            n=len(up_streaks)
        ))

        step.stats.append(StatResult(
            name="Down 連勝平均長度",
            value=down_streaks.mean(),
            n=len(down_streaks)
        ))

        step.stats.append(StatResult(
            name="Down 連勝最長",
            value=down_streaks.max(),
            n=len(down_streaks)
        ))

        # Conditional on streak length
        results_by_streak = []
        for streak_len in [2, 3, 5]:
            # After N consecutive Ups
            cols = [f"prev_{i}" for i in range(1, streak_len + 1)]
            mask_n_ups = (df[cols].sum(axis=1) == streak_len)
            if mask_n_ups.sum() >= 30:
                rate = df.loc[mask_n_ups, "is_up"].mean()
                results_by_streak.append((f"P(Up | {streak_len}連Up)", rate, mask_n_ups.sum()))

            # After N consecutive Downs
            mask_n_downs = (df[cols].sum(axis=1) == 0) & df[cols].notna().all(axis=1)
            if mask_n_downs.sum() >= 30:
                rate = df.loc[mask_n_downs, "is_up"].mean()
                results_by_streak.append((f"P(Up | {streak_len}連Down)", rate, mask_n_downs.sum()))

        for name, rate, n in results_by_streak:
            step.stats.append(StatResult(name=name, value=rate, n=int(n)))

        # Findings
        if p_momentum < 0.05:
            if after_up_rate > after_down_rate:
                step.findings.append(f"存在動量效應 - Up 後更易 Up ({after_up_rate:.1%} vs {after_down_rate:.1%})")
            else:
                step.findings.append(f"存在均值回歸 - Up 後更易 Down ({after_up_rate:.1%} vs {after_down_rate:.1%})")
        else:
            step.findings.append("無顯著連續性模式 - 接近獨立分布")

        # Expected streak length for fair coin
        expected_streak = 2.0  # For p=0.5, expected run length is 2
        if abs(up_streaks.mean() - expected_streak) > 0.3:
            step.findings.append(f"連勝長度偏離隨機預期 (實際:{up_streaks.mean():.2f} vs 預期:{expected_streak:.1f})")

        self.steps.append(step)
        return step

    def step4_cross_comparison(self) -> AnalysisStep:
        """Step 4: 橫向對比"""
        step = AnalysisStep(4, "橫向對比 (Cross Comparisons)")
        df = self.df.copy()

        # Trading sessions (UTC)
        # Asia: 00-08, Europe: 08-16, US: 16-24
        def get_session(hour):
            if 0 <= hour < 8:
                return "Asia"
            elif 8 <= hour < 16:
                return "Europe"
            else:
                return "US"

        df["session"] = df["hour"].apply(get_session)
        session_stats = df.groupby("session")["is_up"].agg(["sum", "count", "mean"])
        session_stats.columns = ["up_count", "total", "up_rate"]

        for session in ["Asia", "Europe", "US"]:
            if session in session_stats.index:
                step.stats.append(StatResult(
                    name=f"{session} Session Up 率",
                    value=session_stats.loc[session, "up_rate"],
                    n=int(session_stats.loc[session, "total"])
                ))

        # Chi-square for sessions
        chi2_session, p_session = stats.chisquare(
            session_stats["up_count"],
            f_exp=session_stats["total"] * df["is_up"].mean()
        )

        step.stats.append(StatResult(
            name="交易時段獨立性檢驗",
            value=chi2_session,
            p_value=p_session,
            significant=p_session < 0.05
        ))

        # Weekday vs Weekend
        df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
        weekday_rate = df[df["is_weekend"] == 0]["is_up"].mean()
        weekend_rate = df[df["is_weekend"] == 1]["is_up"].mean()
        weekday_n = (df["is_weekend"] == 0).sum()
        weekend_n = (df["is_weekend"] == 1).sum()

        step.stats.append(StatResult(
            name="Weekday Up 率",
            value=weekday_rate,
            n=int(weekday_n)
        ))

        step.stats.append(StatResult(
            name="Weekend Up 率",
            value=weekend_rate,
            n=int(weekend_n)
        ))

        # Two proportion z-test (manual calculation)
        p1 = weekday_rate
        p2 = weekend_rate
        p_pooled = (df[df["is_weekend"]==0]["is_up"].sum() + df[df["is_weekend"]==1]["is_up"].sum()) / (weekday_n + weekend_n)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/weekday_n + 1/weekend_n))
        z_stat = (p1 - p2) / se if se > 0 else 0
        p_weekend = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        step.stats.append(StatResult(
            name="Weekday vs Weekend 差異",
            value=weekday_rate - weekend_rate,
            p_value=p_weekend,
            significant=p_weekend < 0.05
        ))

        # Volume correlation (if available)
        if df["volume"].notna().any() and df["volume"].std() > 0:
            df["high_volume"] = df["volume"] > df["volume"].median()
            high_vol_rate = df[df["high_volume"]]["is_up"].mean()
            low_vol_rate = df[~df["high_volume"]]["is_up"].mean()

            step.stats.append(StatResult(
                name="高成交量 Up 率",
                value=high_vol_rate,
                n=int(df["high_volume"].sum())
            ))

            step.stats.append(StatResult(
                name="低成交量 Up 率",
                value=low_vol_rate,
                n=int((~df["high_volume"]).sum())
            ))

        # Findings
        if p_session < 0.05:
            best_session = session_stats["up_rate"].idxmax()
            step.findings.append(f"交易時段存在差異 - {best_session} 最有利 ({session_stats.loc[best_session, 'up_rate']:.1%})")
        else:
            step.findings.append("交易時段無顯著差異")

        if p_weekend < 0.05:
            if weekday_rate > weekend_rate:
                step.findings.append(f"Weekday Up 率顯著較高 ({weekday_rate:.1%} vs {weekend_rate:.1%})")
            else:
                step.findings.append(f"Weekend Up 率顯著較高 ({weekend_rate:.1%} vs {weekday_rate:.1%})")
        else:
            step.findings.append("Weekday/Weekend 無顯著差異")

        self.steps.append(step)
        return step

    def step5_statistical_validation(self) -> AnalysisStep:
        """Step 5: 統計顯著性驗證"""
        step = AnalysisStep(5, "統計顯著性驗證 (Statistical Validation)")
        df = self.df

        # Runs test for randomness
        n_up = df["is_up"].sum()
        n_down = len(df) - n_up

        # Count runs
        runs = 1
        for i in range(1, len(df)):
            if df.iloc[i]["is_up"] != df.iloc[i-1]["is_up"]:
                runs += 1

        # Expected runs and variance for random sequence
        n = len(df)
        expected_runs = (2 * n_up * n_down / n) + 1
        var_runs = (2 * n_up * n_down * (2 * n_up * n_down - n)) / (n**2 * (n - 1))
        z_runs = (runs - expected_runs) / np.sqrt(var_runs) if var_runs > 0 else 0
        p_runs = 2 * (1 - stats.norm.cdf(abs(z_runs)))

        step.stats.append(StatResult(
            name="實際 Runs 數",
            value=runs,
            n=n
        ))

        step.stats.append(StatResult(
            name="預期 Runs 數 (隨機)",
            value=expected_runs,
            n=n
        ))

        step.stats.append(StatResult(
            name="Runs Test Z-score",
            value=z_runs,
            p_value=p_runs,
            significant=p_runs < 0.05
        ))

        # Autocorrelation
        is_up_centered = df["is_up"] - df["is_up"].mean()
        autocorr_1 = is_up_centered.autocorr(lag=1)
        autocorr_5 = is_up_centered.autocorr(lag=5)

        step.stats.append(StatResult(
            name="自相關 (lag=1)",
            value=autocorr_1 if not pd.isna(autocorr_1) else 0,
            n=n
        ))

        step.stats.append(StatResult(
            name="自相關 (lag=5)",
            value=autocorr_5 if not pd.isna(autocorr_5) else 0,
            n=n
        ))

        # Ljung-Box test for autocorrelation
        from scipy.stats import chi2
        # Manual Ljung-Box for lag 1
        n_obs = len(df)
        r1 = autocorr_1 if not pd.isna(autocorr_1) else 0
        lb_stat = n_obs * (n_obs + 2) * (r1**2 / (n_obs - 1))
        lb_p = 1 - chi2.cdf(lb_stat, df=1)

        step.stats.append(StatResult(
            name="Ljung-Box 檢驗 (lag=1)",
            value=lb_stat,
            p_value=lb_p,
            significant=lb_p < 0.05
        ))

        # Time stability - split into periods
        n_periods = 4
        period_size = len(df) // n_periods
        period_rates = []
        for i in range(n_periods):
            start = i * period_size
            end = (i + 1) * period_size if i < n_periods - 1 else len(df)
            rate = df.iloc[start:end]["is_up"].mean()
            period_rates.append(rate)
            step.stats.append(StatResult(
                name=f"Period {i+1}/{n_periods} Up 率",
                value=rate,
                n=end - start
            ))

        # Test for trend
        period_trend = stats.pearsonr(range(n_periods), period_rates)
        step.stats.append(StatResult(
            name="時期趨勢相關",
            value=period_trend[0],
            p_value=period_trend[1],
            significant=period_trend[1] < 0.05
        ))

        # Findings
        if p_runs < 0.05:
            if runs < expected_runs:
                step.findings.append(f"Runs 偏少 - 存在群聚效應 (Z={z_runs:.2f})")
            else:
                step.findings.append(f"Runs 偏多 - 存在過度交替 (Z={z_runs:.2f})")
        else:
            step.findings.append("Runs Test 通過 - 序列接近隨機")

        if lb_p < 0.05:
            step.findings.append(f"存在顯著自相關 - 非獨立序列")
        else:
            step.findings.append("無顯著自相關 - 接近獨立序列")

        if period_trend[1] < 0.05:
            direction = "上升" if period_trend[0] > 0 else "下降"
            step.findings.append(f"Up 率有{direction}趨勢 (r={period_trend[0]:.3f})")
        else:
            step.findings.append("Up 率時間穩定 - 無明顯趨勢")

        self.steps.append(step)
        return step

    def step6_insights(self) -> AnalysisStep:
        """Step 6: 總結洞見"""
        step = AnalysisStep(6, "總結洞見與方向 (Insights & Directions)")
        df = self.df

        # Collect significant findings
        significant_patterns = []
        edge_opportunities = []

        for prev_step in self.steps:
            for stat in prev_step.stats:
                if stat.significant:
                    significant_patterns.append(f"{prev_step.title}: {stat.name}")

        # Summary stats
        up_rate = df["is_up"].mean()
        step.stats.append(StatResult(
            name="Overall Up 率",
            value=up_rate,
            n=len(df)
        ))

        # Calculate theoretical edge
        # If betting Up at fair odds (0.5), edge = actual_rate - 0.5
        edge_up = up_rate - 0.5
        edge_down = 0.5 - up_rate

        step.stats.append(StatResult(
            name="理論 Edge (買 Up)",
            value=edge_up * 100,  # in percentage
            n=len(df)
        ))

        step.stats.append(StatResult(
            name="理論 Edge (買 Down)",
            value=edge_down * 100,
            n=len(df)
        ))

        # Kelly criterion (simplified)
        p = up_rate
        q = 1 - p
        b = 1  # Even odds
        kelly_up = (b * p - q) / b if p > 0.5 else 0
        kelly_down = (b * q - p) / b if p < 0.5 else 0

        step.stats.append(StatResult(
            name="Kelly 比例 (買 Up)",
            value=kelly_up,
            n=len(df)
        ))

        step.stats.append(StatResult(
            name="Kelly 比例 (買 Down)",
            value=kelly_down,
            n=len(df)
        ))

        # Findings
        step.findings.append(f"分析 {len(df):,} 個已結算市場")

        if significant_patterns:
            step.findings.append(f"發現 {len(significant_patterns)} 個顯著模式:")
            for pattern in significant_patterns[:5]:
                step.findings.append(f"  • {pattern}")
        else:
            step.findings.append("未發現顯著統計優勢 - 市場接近有效")

        # Direction recommendations
        if abs(up_rate - 0.5) > 0.02:
            direction = "Up" if up_rate > 0.5 else "Down"
            step.findings.append(f"[方向] 整體傾向 {direction} ({up_rate:.1%})")
        else:
            step.findings.append("[方向] 整體接近 50/50 - 無明顯方向偏好")

        # Edge summary
        max_edge = max(abs(edge_up), abs(edge_down)) * 100
        if max_edge < 1:
            step.findings.append("[Edge] 理論 Edge < 1% - 扣除費用後可能無利可圖")
        elif max_edge < 3:
            step.findings.append(f"[Edge] 理論 Edge ≈ {max_edge:.1f}% - 需要大量注數")
        else:
            step.findings.append(f"[Edge] 理論 Edge ≈ {max_edge:.1f}% - 可能存在機會")

        self.steps.append(step)
        return step

    def run_all(self) -> list[AnalysisStep]:
        """執行所有分析步驟"""
        self.load_data()
        self.step1_basic_stats()
        self.step2_time_analysis()
        self.step3_momentum_analysis()
        self.step4_cross_comparison()
        self.step5_statistical_validation()
        self.step6_insights()
        return self.steps

    def print_report(self) -> str:
        """生成完整報告"""
        lines = [
            "=" * 70,
            "BTC Up/Down 5m Market 概率統計分析報告",
            "=" * 70,
        ]

        for step in self.steps:
            lines.append(step.summary())

        lines.append("\n" + "=" * 70)
        lines.append("END OF REPORT")
        lines.append("=" * 70)

        return "\n".join(lines)


def main():
    """Main entry point"""
    analysis = BTCUpDownProbabilityAnalysis()
    analysis.run_all()
    print(analysis.print_report())
    return analysis


if __name__ == "__main__":
    main()
