# Architecture Specification: StatArb ML Exit & Risk Manager

> **Errata applied on save:** engine time-stop uses
> `max(5, ⌈2.5 × half_life⌉)` (floor of 5 bars), **not** `min(5, …)`.
> `min` would always cap holds at 5 bars and defeat half-life scaling.
> Config key renamed to `absolute_min_bars` to match.

## 1. System Overview

**Target:** Upgrade the existing statistical arbitrage execution engine by replacing
static exit triggers with a high-precision ML exit pipeline paired with
deterministic risk controls (based on Sarmento & Horta, 2020).

```
┌─────────────────────────────────────────────────────────┐
│                    Market Data Input                    │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Kalman Filter State Estimator              │
│       Computes dynamic beta, spread z-score, vol        │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│               Engine Risk Safeguards                    │
│   Hard Time-Stop: bars_held >= max(5, 2.5 * half_life)  │
│   Hard Stop-Loss / Structural De-integration            │
└──────────────┬───────────────────────────┬──────────────┘
               │ Triggers Stop             │ Passes
               ▼                           ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│   FORCE CLOSE POSITION   │  │   ML Exit Evaluator      │
└──────────────────────────┘  │   Logistic Probability   │
                              └────────────┬─────────────┘
                                           │
                                           ▼
                              ┌──────────────────────────┐
                              │ Prob >= exit_threshold   │
                              │ (Default: 0.68)          │
                              └────────────┬─────────────┘
                                           │
                                  Yes ┌────┴────┐ No
                                      ▼         ▼
                                   [EXIT]    [HOLD]
```

## 2. Data & Feature Engineering (`src/features.py`)

### 2.1 Feature Set Clean-Up

To eliminate collinearity and enforce long/short direction symmetry, prune the
input feature vector down to 8 uncorrelated features.

| Feature Name | Type | Description | Handling Rule |
|---|---|---|---|
| `vol` | float | Rolling spread volatility | Keep |
| `pnl_proxy` | float | Cumulative unrealized PnL in z-units | Keep |
| `abs_entry_z` | float | Absolute value $\|z_{\text{entry}}\|$ at trade open | Keep (enforces symmetry) |
| `confidence` | float | Kalman filter state estimation confidence | Keep |
| `exit_z` | float | Current bar spread z-score | Keep |
| `velocity` | float | Rate of z-score change over last 3 bars | Keep |
| `bars_held` | int | Number of bars since entry | Keep |
| `half_life` | float | OU mean-reversion half-life (normalized units) | Keep |
| `entry_z` | — | Directional entry z-score | **DROP** (directional bias) |
| `favorable` | — | Unrealized max gain | **DROP** (duplicate of `pnl_proxy`) |
| `best_fav` | — | High water mark unrealized gain | **DROP** (duplicate of `pnl_proxy`) |

### 2.2 Feature Normalization Pipeline

* Fit a `StandardScaler` strictly on historical training data for the 8 selected features.
* Serialize the fitted scaler object to `models/feature_scaler.pkl`.
* At runtime, transform incoming trade dict vectors through `scaler.transform()`
  before feeding them to the logit function.

## 3. Offline Labeling & Training Pipeline (`src/train_exit_model.py`)

### 3.1 Binary Target Labeling ($y$)

For each historical execution log bar $t$ during an active trade:

* Set lookahead horizon $H = \lceil 2.5 \times \text{half_life_bars} \rceil$.
* Calculate future net profit:
  $\Delta \text{PnL}_{\text{net}} = \text{PnL}_{t+h} - \text{Transaction Costs}$.
* Set $y_t = 1$ if $\Delta \text{PnL}_{\text{net}} > 0$ within $h \le H$ bars
  and max adverse excursion stays within the stop-loss limit.
* Set $y_t = 0$ otherwise.

### 3.2 Model Fitting Requirements

* **Model Type:** `sklearn.linear_model.LogisticRegression` with L1 or L2 penalty (`C=1.0`).
* **Calibration:** Apply `CalibratedClassifierCV` (Isotonic or Sigmoid) if raw logit
  probabilities show empirical drift.
* **Artifacts Generated:**
  * `models/logistic_exit_model.pkl` — trained weights & bias
  * `models/feature_scaler.pkl` — fitted `StandardScaler`
  * `models/model_metadata.json` — training date, metrics, feature order

## 4. Production Exit Manager (`src/exit_manager.py`)

Implement `StatArbExitManager` to process real-time execution updates.

```python
from dataclasses import dataclass
import joblib
import numpy as np

@dataclass
class TradeState:
    trade_id: int
    ticker_a: str
    ticker_b: str
    direction: str
    bars_held: int
    half_life_bars: float
    vol: float
    pnl_proxy: float
    entry_z: float
    confidence: float
    exit_z: float
    velocity: float
    half_life: float
    cost_dollars: float

class StatArbExitManager:
    def __init__(self, model_path: str, scaler_path: str, exit_threshold: float = 0.68):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.exit_threshold = exit_threshold
        self.feature_names = [
            "vol", "pnl_proxy", "abs_entry_z", "confidence",
            "exit_z", "velocity", "bars_held", "half_life"
        ]

    def evaluate_trade(self, state: TradeState) -> tuple[bool, str, float]:
        # 1. Deterministic Time-Stop Guard (floor 5 bars, then scale with HL)
        max_bars = max(5, int(np.ceil(2.5 * state.half_life_bars)))
        if state.bars_held >= max_bars:
            return True, f"TIME_STOP_EXPIRED (Held {state.bars_held} >= {max_bars} bars)", 1.0

        # 2. Extract & Format Features
        raw_features = np.array([[
            state.vol,
            state.pnl_proxy,
            abs(state.entry_z),  # Force absolute value for symmetry
            state.confidence,
            state.exit_z,
            state.velocity,
            state.bars_held,
            state.half_life
        ]])

        # 3. Scale & Predict
        scaled_features = self.scaler.transform(raw_features)
        prob_exit = float(self.model.predict_proba(scaled_features)[0][1])

        # 4. Threshold Check
        if prob_exit >= self.exit_threshold:
            return True, f"ML_EXIT_TRIGGERED (Prob {prob_exit:.3f} >= {self.exit_threshold})", prob_exit

        return False, f"HOLD (Prob {prob_exit:.3f} < {self.exit_threshold})", prob_exit
```

## 5. System Configuration (`config/strategy_config.yaml`)

```yaml
strategy:
  name: "kalman_statarb_pairs"
  version: "2.1.0"

exit_model:
  model_path: "models/logistic_exit_model.pkl"
  scaler_path: "models/feature_scaler.pkl"
  probability_threshold: 0.68

risk_engine:
  max_half_life_multiplier: 2.5
  absolute_min_bars: 5          # floor for time-stop (NOT a cap)
  stop_loss_z: 4.0
  hard_pnl_stop_dollars: -150.00

execution:
  broker: "alpaca_paper"  # or alpaca_live
  order_type: "market"
```

## 6. Action Plan for Cursor AI

Prompt Cursor with:

> Build the file structure outlined in `SYSTEM_SPEC.md`. Implement
> `src/features.py`, `src/train_exit_model.py`, `src/exit_manager.py`, and
> `config/strategy_config.yaml`. Ensure feature extraction handles long/short
> symmetry by forcing `abs_entry_z`, drops `favorable`, `best_fav`, and
> `entry_z`, and wraps evaluation in type-checked dataclasses with unit tests
> in `tests/test_exit_manager.py`.

### Alignment with current `main` (post exit-invariants)

| Spec item | Status |
|---|---|
| Drop `favorable` / `best_fav` / raw `entry_z` | Done (schema v3, `abs_entry_z`) |
| Symmetry via absolute entry depth | Done as `abs_entry_z` |
| Default threshold 0.68 | Done (`config/strategy_config.yaml` + `src/config.py`) |
| Time-stop `max(5, 2.5×hl)` | Done in `StatArbExitManager` / `time_stop_bars` |
| Z-score / standardize features | Done via `StandardScaler` → `models/feature_scaler.pkl` |
| Split `src/` + sklearn `joblib` artifacts + YAML config | Done — `src/{features,exit_manager,train_exit_model,config,kalman,journal}`; monolith still hosts CLI/loop |
| Fail-closed without `StatArbExitManager` | Done — no live/backtest/research fallback to JSON logistic |
| Training floor `min_samples≥50` | Done — config + `train_exit_model` CLI |
| NaN-safe ML path | Done — `evaluate_trade` holds on non-finite features |
| Lookahead labeling $H = 2.5×hl$ | Done in `src/train_exit_model.label_path_bars` |
| Legacy `LogisticExitModel` | Quarantined in `src/legacy_logistic.py` (tests / old JSON only) |
| Config-driven entry (`z_entry` / `min_confidence`) + Kalman knobs | Done — `entry_direction` / `build_kalman` read YAML |
| Soft MR bands from `risk_engine.soft_mr_*` | Done — on manager; used when weights missing |
| AR(1) half-life estimator (`src/half_life.py`) | Done — lookback=80, floor 4 bars (was collapsing to 2) |
| Path-label harvest (`src/harvest.py`, `--harvest-training`) | Done — per-bar `label_path_bars` → dataset |
| Training `class_weight=balanced` + pair holdout + coef metadata | Done — `src/train_exit_model.py` / YAML |
| Further monolith split (`PositionState`, broker loop) | Deferred |

