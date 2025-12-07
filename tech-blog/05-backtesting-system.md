# Qlib回测系统深度解析：构建专业级量化回测引擎

## 引言

回测系统是量化投资的实验室，是验证投资策略有效性的关键工具。Qlib提供了专业级的回测框架，支持从简单的策略验证到复杂的风险分析的全流程回测需求。本文将深入分析Qlib回测系统的设计思想、实现原理和最佳实践，帮助读者构建准确、高效的量化回测系统。

## 回测系统架构概览

### 整体架构设计

Qlib回测系统采用了事件驱动的分层架构设计，将复杂的回测过程分解为多个独立但协作的层次：

```
┌─────────────────────────────────────────────────────────────┐
│                    应用层 (Application Layer)                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   策略回测      │  │   性能分析      │  │   风险评估      │  │
│  │Strategy Backtest│  │Performance Anal │  │  Risk Analysis  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    决策层 (Decision Layer)                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   策略决策      │  │   订单生成      │  │   仓位管理      │  │
│  │Strategy Decision│  │ Order Generator │  │Position Management│  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    执行层 (Execution Layer)                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   交易所模拟    │  │   订单执行      │  │   成本计算      │  │
│  │Exchange Simulation│  │ Order Execution │  │ Cost Calculation│  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    账户层 (Account Layer)                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   账户管理      │  │   头寸跟踪      │  │   资金流水      │  │
│  │Account Mgmt     │  │Position Tracking│  │Cash Flow Track  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    数据层 (Data Layer)                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   市场数据      │  │   交易日历      │  │   基准数据      │  │
│  │ Market Data     │  │Trading Calendar │  │Benchmark Data   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件关系

```python
# 回测系统核心组件
Backtesting Architecture:
├── Executor (执行器)
│   ├── BaseExecutor (基础执行器)
│   ├── NestedExecutor (嵌套执行器)
│   └── SimulatorExecutor (模拟执行器)
├── Strategy (策略)
│   ├── BaseStrategy (策略基类)
│   ├── SignalStrategy (信号策略)
│   └── PortfolioStrategy (组合策略)
├── Exchange (交易所)
│   ├── BaseExchange (交易所基类)
│   ├── Exchange (模拟交易所)
│   └── Order (订单对象)
├── Account (账户)
│   ├── Account (账户管理)
│   └── Position (头寸管理)
└── Analysis (分析)
    ├── AnalysisEngine (分析引擎)
    ├── Report (报告生成)
    └── Attribution (归因分析)
```

## 核心组件深度解析

### 回测执行器设计

回测执行器是整个回测系统的核心，负责协调策略、交易所和账户之间的交互：

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from qlib.strategy.base import BaseStrategy
from qlib.backtest.exchange import Exchange
from qlib.backtest.account import Account
from qlib.backtest.decision import BaseTradeDecision

class BaseExecutor(ABC):
    """
    回测执行器基类

    设计理念：
    1. 定义标准的执行流程接口
    2. 支持多种执行模式
    3. 提供灵活的配置选项
    4. 统一的交易记录和报告机制
    """

    def __init__(self, time_per_step="day", generate_portfolio_metrics=True, **kwargs):
        """
        初始化执行器

        Args:
            time_per_step: 交易步长 ("day", "hour", "min")
            generate_portfolio_metrics: 是否生成投资组合指标
            **kwargs: 其他配置参数
        """
        self.time_per_step = time_per_step
        self.generate_portfolio_metrics = generate_portfolio_metrics
        self.trade_calendar = None
        self.trade_account = None
        self.trade_exchange = None

    @abstractmethod
    def execute(self, strategy: BaseStrategy, **kwargs) -> Dict[str, Any]:
        """
        执行回测的抽象方法

        Args:
            strategy: 交易策略
            **kwargs: 其他执行参数

        Returns:
            回测结果字典
        """
        raise NotImplementedError("Subclasses must implement execute method")

    def _initialize_components(self, strategy: BaseStrategy, **kwargs):
        """初始化回测组件"""
        # 1. 初始化交易日历
        self.trade_calendar = kwargs.get("trade_calendar", self._get_default_calendar())

        # 2. 初始化交易所
        if self.trade_exchange is None:
            self.trade_exchange = Exchange(**kwargs.get("exchange_kwargs", {}))

        # 3. 初始化账户
        if self.trade_account is None:
            account_kwargs = kwargs.get("account_kwargs", {})
            account_kwargs["init_cash"] = account_kwargs.get("init_cash", 1e8)  # 默认1亿
            self.trade_account = Account(**account_kwargs)

    def _get_default_calendar(self):
        """获取默认交易日历"""
        from qlib.data import D
        return D.calendar()

class NestedExecutor(BaseExecutor):
    """
    嵌套执行器

    特性：
    1. 支持多层嵌套的回测结构
    2. 用于复杂的资金管理和多策略回测
    3. 提供灵活的资源分配机制
    4. 支持策略间的资金调配
    """

    def __init__(self, outer_executor=None, inner_executor=None,
                 allocate_ratio=1.0, **kwargs):
        """
        初始化嵌套执行器

        Args:
            outer_executor: 外层执行器
            inner_executor: 内层执行器
            allocate_ratio: 资金分配比例
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.outer_executor = outer_executor
        self.inner_executor = inner_executor
        self.allocate_ratio = allocate_ratio

    def execute(self, strategy: BaseStrategy, **kwargs):
        """执行嵌套回测"""
        # 1. 初始化组件
        self._initialize_components(strategy, **kwargs)

        # 2. 外层执行逻辑
        if self.outer_executor:
            outer_results = self.outer_executor.execute(strategy, **kwargs)
        else:
            outer_results = {}

        # 3. 内层执行逻辑
        if self.inner_executor:
            # 调整内层执行器的资金
            inner_kwargs = kwargs.copy()
            account_kwargs = inner_kwargs.get("account_kwargs", {})
            account_kwargs["init_cash"] = account_kwargs.get("init_cash", 1e8) * self.allocate_ratio
            inner_kwargs["account_kwargs"] = account_kwargs

            inner_results = self.inner_executor.execute(strategy, **inner_kwargs)
        else:
            inner_results = {}

        # 4. 合并结果
        return {
            "outer": outer_results,
            "inner": inner_results,
            "combined": self._combine_results(outer_results, inner_results)
        }

    def _combine_results(self, outer_results: Dict, inner_results: Dict) -> Dict:
        """合并内外层回测结果"""
        combined = {}

        # 合并收益率曲线
        if "portfolio_analysis" in outer_results and "portfolio_analysis" in inner_results:
            outer_pa = outer_results["portfolio_analysis"]
            inner_pa = inner_results["portfolio_analysis"]

            # 按权重合并收益率
            combined_returns = (
                outer_pa.get("return_curve", pd.Series()) * (1 - self.allocate_ratio) +
                inner_pa.get("return_curve", pd.Series()) * self.allocate_ratio
            )

            combined["portfolio_analysis"] = {
                "return_curve": combined_returns,
                # 其他合并逻辑...
            }

        return combined

class SimulatorExecutor(BaseExecutor):
    """
    模拟执行器

    特性：
    1. 实现完整的回测模拟流程
    2. 支持事件驱动的交易执行
    3. 提供详细的交易记录和分析
    4. 支持多种交易成本模型
    """

    def __init__(self, verbose=True, **kwargs):
        """
        初始化模拟执行器

        Args:
            verbose: 是否输出详细日志
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.verbose = verbose
        self.trade_steps = []
        self.trade_records = []

    def execute(self, strategy: BaseStrategy, **kwargs) -> Dict[str, Any]:
        """执行回测模拟"""
        # 1. 初始化组件
        self._initialize_components(strategy, **kwargs)

        # 2. 获取交易时间范围
        start_time = kwargs.get("start_time")
        end_time = kwargs.get("end_time")
        if not start_time or not end_time:
            raise ValueError("start_time and end_time must be provided")

        # 3. 生成交易步骤
        self.trade_steps = self._generate_trade_steps(start_time, end_time)

        # 4. 主回测循环
        for i, trade_step in enumerate(self.trade_steps):
            if self.verbose:
                print(f"Trading step {i+1}/{len(self.trade_steps)}: {trade_step}")

            # 4.1 获取市场数据
            market_data = self._get_market_data(trade_step)

            # 4.2 策略生成交易决策
            trade_decision = self._generate_strategy_decision(strategy, market_data, trade_step)

            # 4.3 执行交易
            execution_result = self._execute_trade_decision(trade_decision, trade_step)

            # 4.4 更新账户状态
            self._update_account(execution_result, trade_step)

            # 4.5 记录交易步骤
            self._record_trade_step(trade_step, trade_decision, execution_result)

        # 5. 生成回测报告
        return self._generate_backtest_report()

    def _generate_trade_steps(self, start_time: str, end_time: str) -> List[pd.Timestamp]:
        """生成交易步骤"""
        # 1. 获取交易日历
        trade_dates = self.trade_calendar

        # 2. 过滤交易时间范围
        start_ts = pd.Timestamp(start_time)
        end_ts = pd.Timestamp(end_time)

        valid_dates = [date for date in trade_dates if start_ts <= date <= end_ts]

        # 3. 根据时间步长调整
        if self.time_per_step == "day":
            return valid_dates
        elif self.time_per_step == "hour":
            # 扩展为小时级别（概念性实现）
            expanded_steps = []
            for date in valid_dates:
                for hour in range(9, 16):  # 9:00-15:00
                    expanded_steps.append(pd.Timestamp(f"{date} {hour:02d}:00"))
            return expanded_steps
        else:
            raise ValueError(f"Unsupported time_per_step: {self.time_per_step}")

    def _get_market_data(self, trade_step: pd.Timestamp) -> Dict[str, pd.DataFrame]:
        """获取市场数据"""
        from qlib.data import D

        # 1. 获取基础行情数据
        instruments = self.trade_account.get_current_holdings().keys()
        if not instruments:
            # 如果没有持仓，获取市场基准股票
            instruments = D.instruments(market="csi300")[:100]

        # 2. 获取OHLCV数据
        market_data = {}
        try:
            price_data = D.features(
                instruments,
                ['$close', '$open', '$high', '$low', '$volume'],
                start_time=trade_step.strftime("%Y-%m-%d"),
                end_time=trade_step.strftime("%Y-%m-%d"),
                freq="day"
            )
            market_data["price"] = price_data
        except Exception as e:
            if self.verbose:
                print(f"Warning: Failed to get market data for {trade_step}: {e}")
            market_data["price"] = pd.DataFrame()

        return market_data

    def _generate_strategy_decision(self, strategy: BaseStrategy,
                                   market_data: Dict, trade_step: pd.Timestamp) -> BaseTradeDecision:
        """生成策略决策"""
        try:
            # 1. 准备策略输入数据
            strategy_input = {
                "time": trade_step,
                "market": market_data,
                "account": self.trade_account.get_account_info()
            }

            # 2. 调用策略生成决策
            trade_decision = strategy.generate_trade_decision(
                trade_step=trade_step,
                account=self.trade_account,
                exchange=self.trade_exchange,
                **strategy_input
            )

            return trade_decision

        except Exception as e:
            if self.verbose:
                print(f"Warning: Strategy decision generation failed for {trade_step}: {e}")
            # 返回空决策
            from qlib.backtest.decision import EmptyTradeDecision
            return EmptyTradeDecision()

    def _execute_trade_decision(self, trade_decision: BaseTradeDecision,
                               trade_step: pd.Timestamp) -> Dict[str, Any]:
        """执行交易决策"""
        execution_result = {
            "trade_step": trade_step,
            "orders": [],
            "trades": [],
            "execution_cost": 0.0
        }

        try:
            # 1. 获取交易订单
            orders = trade_decision.get_order_list()
            execution_result["orders"] = orders

            # 2. 在交易所执行订单
            if orders:
                trades = self.trade_exchange.deal_order(orders, trade_step)
                execution_result["trades"] = trades

                # 3. 计算交易成本
                execution_result["execution_cost"] = self._calculate_execution_cost(trades)

        except Exception as e:
            if self.verbose:
                print(f"Warning: Trade execution failed for {trade_step}: {e}")

        return execution_result

    def _update_account(self, execution_result: Dict, trade_step: pd.Timestamp):
        """更新账户状态"""
        try:
            # 1. 更新持仓
            self.trade_account.update_position(execution_result["trades"], trade_step)

            # 2. 更新资金
            self.trade_account.update_cash(execution_result["execution_cost"])

            # 3. 计算当前资产价值
            self.trade_account.update_portfolio_value(trade_step)

        except Exception as e:
            if self.verbose:
                print(f"Warning: Account update failed for {trade_step}: {e}")

    def _calculate_execution_cost(self, trades: List) -> float:
        """计算交易成本"""
        total_cost = 0.0

        for trade in trades:
            # 1. 计算手续费
            commission = self.trade_exchange.get_commission(trade)
            total_cost += commission

            # 2. 计算印花税（仅卖出）
            if trade.direction < 0:  # 卖出
                tax = self.trade_exchange.get_tax(trade)
                total_cost += tax

        return total_cost

    def _record_trade_step(self, trade_step: pd.Timestamp,
                          trade_decision: BaseTradeDecision,
                          execution_result: Dict):
        """记录交易步骤"""
        step_record = {
            "datetime": trade_step,
            "decision_type": type(trade_decision).__name__,
            "order_count": len(execution_result["orders"]),
            "trade_count": len(execution_result["trades"]),
            "execution_cost": execution_result["execution_cost"],
            "portfolio_value": self.trade_account.get_total_value()
        }

        self.trade_records.append(step_record)

    def _generate_backtest_report(self) -> Dict[str, Any]:
        """生成回测报告"""
        from qlib.contrib.evaluate import risk_analysis
        from qlib.contrib.strategy import BaseStrategy

        # 1. 基础交易统计
        trade_df = pd.DataFrame(self.trade_records)
        if trade_df.empty:
            return {"error": "No trade records available"}

        # 2. 计算收益率曲线
        portfolio_values = trade_df["portfolio_value"]
        return_curve = portfolio_values.pct_change().fillna(0)
        cumulative_return = (1 + return_curve).cumprod() - 1

        # 3. 风险分析
        analysis_result = risk_analysis(cumulative_return)

        # 4. 交易统计
        trade_stats = {
            "total_trades": len(self.trade_records),
            "profitable_trades": len(trade_df[trade_df["portfolio_value"].diff() > 0]),
            "total_execution_cost": trade_df["execution_cost"].sum(),
            "average_daily_return": return_curve.mean(),
            "return_volatility": return_curve.std(),
            "sharpe_ratio": return_curve.mean() / (return_curve.std() + 1e-10) * np.sqrt(252),
            "max_drawdown": analysis_result.get("max_drawdown", 0),
            "total_return": cumulative_return.iloc[-1] if len(cumulative_return) > 0 else 0
        }

        # 5. 构建报告
        report = {
            "trade_records": trade_df,
            "return_curve": cumulative_return,
            "trade_statistics": trade_stats,
            "risk_analysis": analysis_result,
            "portfolio_analysis": {
                "final_value": portfolio_values.iloc[-1] if len(portfolio_values) > 0 else 0,
                "total_return": trade_stats["total_return"],
                "sharpe_ratio": trade_stats["sharpe_ratio"],
                "max_drawdown": trade_stats["max_drawdown"]
            }
        }

        return report
```

### 交易所模拟系统

交易所模拟系统是回测的核心组件，负责处理订单执行和成本计算：

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from decimal import Decimal
import pandas as pd
import numpy as np
from qlib.backtest.order import Order, Trade

class BaseExchange(ABC):
    """交易所基类"""

    @abstractmethod
    def deal_order(self, orders: List[Order], current_time: pd.Timestamp) -> List[Trade]:
        """处理订单"""
        pass

    @abstractmethod
    def get_commission(self, trade: Trade) -> float:
        """计算手续费"""
        pass

    @abstractmethod
    def get_tax(self, trade: Trade) -> float:
        """计算税费"""
        pass

class Exchange(BaseExchange):
    """
    模拟交易所实现

    特性：
    1. 支持多种订单类型
    2. 真实的交易成本建模
    3. 滑点和市场冲击建模
    4. 涨跌停限制处理
    """

    def __init__(self,
                 commission_rate=0.0003,  # 万分之三手续费
                 tax_rate=0.001,           # 千分之一印花税
                 min_commission=5.0,       # 最低手续费5元
                 slippage_rate=0.001,      # 千分之一滑点
                 market_impact_rate=0.0001, # 万分之一市场冲击
                 limit_threshold=0.095,    # 涨跌停限制9.5%
                 **kwargs):
        """
        初始化交易所

        Args:
            commission_rate: 手续费率
            tax_rate: 印花税率
            min_commission: 最低手续费
            slippage_rate: 滑点率
            market_impact_rate: 市场冲击率
            limit_threshold: 涨跌停阈值
            **kwargs: 其他参数
        """
        self.commission_rate = commission_rate
        self.tax_rate = tax_rate
        self.min_commission = min_commission
        self.slippage_rate = slippage_rate
        self.market_impact_rate = market_impact_rate
        self.limit_threshold = limit_threshold
        self.market_data_cache = {}

    def deal_order(self, orders: List[Order], current_time: pd.Timestamp) -> List[Trade]:
        """
        处理订单

        Args:
            orders: 订单列表
            current_time: 当前时间

        Returns:
            成交列表
        """
        trades = []

        for order in orders:
            try:
                # 1. 获取市场数据
                market_price = self._get_market_price(order.stock_id, current_time)

                if market_price is None:
                    continue  # 无法获取价格，跳过订单

                # 2. 检查涨跌停限制
                if self._check_price_limit(order.stock_id, market_price, current_time):
                    # 触及涨跌停，无法成交
                    continue

                # 3. 计算执行价格（考虑滑点）
                execution_price = self._calculate_execution_price(
                    order, market_price, current_time
                )

                # 4. 计算执行数量（考虑市场冲击）
                execution_amount = self._calculate_execution_amount(
                    order, current_time
                )

                if execution_amount <= 0:
                    continue

                # 5. 创建交易记录
                trade = Trade(
                    stock_id=order.stock_id,
                    amount=execution_amount,
                    direction=order.direction,
                    price=execution_price,
                    datetime=current_time,
                    order_id=order.order_id
                )

                trades.append(trade)

            except Exception as e:
                print(f"Warning: Order execution failed: {e}")
                continue

        return trades

    def _get_market_price(self, stock_id: str, current_time: pd.Timestamp) -> Optional[float]:
        """获取市场价格"""
        try:
            # 1. 检查缓存
            cache_key = f"{stock_id}_{current_time}"
            if cache_key in self.market_data_cache:
                return self.market_data_cache[cache_key]

            # 2. 从数据源获取价格
            from qlib.data import D

            price_data = D.features(
                [stock_id],
                ['$close'],
                start_time=current_time.strftime("%Y-%m-%d"),
                end_time=current_time.strftime("%Y-%m-%d"),
                freq="day"
            )

            if price_data.empty:
                return None

            price = float(price_data.iloc[0, 0])

            # 3. 更新缓存
            self.market_data_cache[cache_key] = price

            return price

        except Exception as e:
            print(f"Warning: Failed to get market price for {stock_id}: {e}")
            return None

    def _check_price_limit(self, stock_id: str, price: float,
                          current_time: pd.Timestamp) -> bool:
        """检查涨跌停限制"""
        try:
            # 1. 获取昨日收盘价
            from qlib.data import D

            prev_date = current_time - pd.Timedelta(days=1)
            prev_price_data = D.features(
                [stock_id],
                ['$close'],
                start_time=prev_date.strftime("%Y-%m-%d"),
                end_time=prev_date.strftime("%Y-%m-%d"),
                freq="day"
            )

            if prev_price_data.empty:
                return False

            prev_price = float(prev_price_data.iloc[0, 0])

            # 2. 计算涨跌停价格
            upper_limit = prev_price * (1 + self.limit_threshold)
            lower_limit = prev_price * (1 - self.limit_threshold)

            # 3. 检查是否触及涨跌停
            return price >= upper_limit or price <= lower_limit

        except Exception:
            return False  # 无法判断，允许交易

    def _calculate_execution_price(self, order: Order, market_price: float,
                                  current_time: pd.Timestamp) -> float:
        """
        计算执行价格（考虑滑点）

        滑点模型：
        - 买入订单：价格上移（不利）
        - 卖出订单：价格下移（不利）
        """
        if order.direction > 0:  # 买入
            slippage_multiplier = 1 + self.slippage_rate
        else:  # 卖出
            slippage_multiplier = 1 - self.slippage_rate

        execution_price = market_price * slippage_multiplier

        # 考虑市场冲击（基于订单金额）
        order_value = abs(order.amount * market_price)
        market_impact = min(self.market_impact_rate * np.log(1 + order_value / 1e6), 0.01)

        if order.direction > 0:  # 买入
            execution_price *= (1 + market_impact)
        else:  # 卖出
            execution_price *= (1 - market_impact)

        return float(execution_price)

    def _calculate_execution_amount(self, order: Order,
                                  current_time: pd.Timestamp) -> int:
        """
        计算执行数量（考虑市场冲击）

        市场冲击模型：
        - 大订单可能无法完全成交
        - 基于成交量和订单大小计算成交率
        """
        try:
            # 1. 获取当日成交量
            from qlib.data import D

            volume_data = D.features(
                [order.stock_id],
                ['$volume'],
                start_time=current_time.strftime("%Y-%m-%d"),
                end_time=current_time.strftime("%Y-%m-%d"),
                freq="day"
            )

            if volume_data.empty:
                return order.amount  # 无法获取成交量，假设全部成交

            daily_volume = float(volume_data.iloc[0, 0])

            # 2. 计算市场冲击率
            order_volume_ratio = abs(order.amount) / max(daily_volume, 1)

            # 3. 基于冲击率计算成交率
            if order_volume_ratio < 0.1:  # 小于10%成交量
                execution_rate = 1.0
            elif order_volume_ratio < 0.3:  # 10%-30%成交量
                execution_rate = 0.8
            elif order_volume_ratio < 0.5:  # 30%-50%成交量
                execution_rate = 0.6
            else:  # 大于50%成交量
                execution_rate = 0.4

            # 4. 计算实际成交数量
            execution_amount = int(order.amount * execution_rate)

            return execution_amount

        except Exception:
            return order.amount  # 计算失败，假设全部成交

    def get_commission(self, trade: Trade) -> float:
        """
        计算手续费

        规则：
        1. 按成交金额的万分之三收取
        2. 最低收费5元
        """
        trade_value = abs(trade.amount * trade.price)
        commission = trade_value * self.commission_rate

        # 最低手续费
        commission = max(commission, self.min_commission)

        return commission

    def get_tax(self, trade: Trade) -> float:
        """
        计算印花税

        规则：
        1. 仅卖出时收取
        2. 按成交金额的千分之一收取
        """
        if trade.direction < 0:  # 卖出
            trade_value = abs(trade.amount * trade.price)
            tax = trade_value * self.tax_rate
            return tax
        else:
            return 0.0

    def get_total_cost(self, trade: Trade) -> float:
        """计算总交易成本"""
        return self.get_commission(trade) + self.get_tax(trade)
```

### 账户和头寸管理

账户管理负责跟踪资金流动、持仓变化和投资组合价值：

```python
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from decimal import Decimal

class Position:
    """
    头寸管理类

    功能：
    1. 跟踪单个股票的持仓
    2. 计算持仓成本和盈亏
    3. 处理分红和除权
    4. 计算持仓收益
    """

    def __init__(self, stock_id: str):
        """
        初始化头寸

        Args:
            stock_id: 股票代码
        """
        self.stock_id = stock_id
        self.amount = 0  # 持仓数量
        self.avg_cost = 0.0  # 平均成本
        self.market_value = 0.0  # 市值
        self.unrealized_pnl = 0.0  # 未实现盈亏
        self.realized_pnl = 0.0  # 已实现盈亏
        self.last_price = 0.0  # 最新价格
        self.transactions = []  # 交易记录

    def add_position(self, amount: int, price: float):
        """
        增加持仓

        Args:
            amount: 增加数量（正数）
            price: 买入价格
        """
        if amount <= 0:
            return

        # 1. 计算新的平均成本
        total_cost_before = self.amount * self.avg_cost
        total_cost_new = amount * price
        total_amount_new = self.amount + amount

        self.avg_cost = (total_cost_before + total_cost_new) / total_amount_new
        self.amount = total_amount_new

        # 2. 记录交易
        self.transactions.append({
            "type": "buy",
            "amount": amount,
            "price": price,
            "timestamp": pd.Timestamp.now()
        })

    def reduce_position(self, amount: int, price: float):
        """
        减少持仓

        Args:
            amount: 减少数量（正数）
            price: 卖出价格
        """
        if amount <= 0 or amount > self.amount:
            return

        # 1. 计算已实现盈亏
        cost_basis = amount * self.avg_cost
        sale_proceeds = amount * price
        realized_pnl = sale_proceeds - cost_basis

        self.realized_pnl += realized_pnl
        self.amount -= amount

        # 2. 如果清仓，重置平均成本
        if self.amount == 0:
            self.avg_cost = 0.0

        # 3. 记录交易
        self.transactions.append({
            "type": "sell",
            "amount": amount,
            "price": price,
            "timestamp": pd.Timestamp.now(),
            "realized_pnl": realized_pnl
        })

    def update_market_value(self, current_price: float):
        """
        更新市值和未实现盈亏

        Args:
            current_price: 当前价格
        """
        self.last_price = current_price
        self.market_value = self.amount * current_price

        if self.amount > 0:
            self.unrealized_pnl = (current_price - self.avg_cost) * self.amount
        else:
            self.unrealized_pnl = 0.0

    def get_position_info(self) -> Dict[str, Any]:
        """获取头寸信息"""
        return {
            "stock_id": self.stock_id,
            "amount": self.amount,
            "avg_cost": self.avg_cost,
            "last_price": self.last_price,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "total_pnl": self.unrealized_pnl + self.realized_pnl,
            "return_pct": (self.last_price / self.avg_cost - 1) if self.avg_cost > 0 else 0.0
        }

class Account:
    """
    账户管理类

    功能：
    1. 管理资金和持仓
    2. 计算投资组合价值
    3. 处理交易和资金流动
    4. 记录账户历史
    """

    def __init__(self, init_cash: float = 1e8):
        """
        初始化账户

        Args:
            init_cash: 初始资金
        """
        self.init_cash = init_cash
        self.current_cash = init_cash
        self.positions = {}  # 持仓字典 {stock_id: Position}
        self.total_value = init_cash  # 总资产
        self.daily_values = []  # 每日资产价值历史
        self.trade_history = []  # 交易历史
        self.current_date = None

    def update_position(self, trades: List, current_time: pd.Timestamp):
        """
        更新持仓

        Args:
            trades: 交易列表
            current_time: 当前时间
        """
        for trade in trades:
            # 1. 确保持仓对象存在
            if trade.stock_id not in self.positions:
                self.positions[trade.stock_id] = Position(trade.stock_id)

            position = self.positions[trade.stock_id]

            # 2. 更新持仓
            if trade.direction > 0:  # 买入
                position.add_position(trade.amount, trade.price)
                self.current_cash -= trade.amount * trade.price
            else:  # 卖出
                position.reduce_position(trade.amount, trade.price)
                self.current_cash += trade.amount * trade.price

            # 3. 记录交易
            self.trade_history.append({
                "datetime": current_time,
                "stock_id": trade.stock_id,
                "amount": trade.amount,
                "price": trade.price,
                "direction": trade.direction,
                "value": abs(trade.amount * trade.price),
                "cash_after": self.current_cash
            })

    def update_cash(self, cost_change: float):
        """
        更新现金（扣除交易成本）

        Args:
            cost_change: 成本变化（负数表示扣除成本）
        """
        self.current_cash += cost_change

    def update_portfolio_value(self, current_time: pd.Timestamp):
        """
        更新投资组合价值

        Args:
            current_time: 当前时间
        """
        # 1. 更新各持仓的市值
        for position in self.positions.values():
            if position.amount > 0:
                current_price = self._get_current_price(position.stock_id, current_time)
                if current_price:
                    position.update_market_value(current_price)

        # 2. 计算总资产价值
        total_position_value = sum(
            pos.market_value for pos in self.positions.values()
        )
        self.total_value = self.current_cash + total_position_value

        # 3. 记录每日价值
        self.daily_values.append({
            "datetime": current_time,
            "total_value": self.total_value,
            "cash": self.current_cash,
            "position_value": total_position_value
        })

        self.current_date = current_time

    def _get_current_price(self, stock_id: str, current_time: pd.Timestamp) -> Optional[float]:
        """获取当前价格"""
        try:
            from qlib.data import D

            price_data = D.features(
                [stock_id],
                ['$close'],
                start_time=current_time.strftime("%Y-%m-%d"),
                end_time=current_time.strftime("%Y-%m-%d"),
                freq="day"
            )

            if not price_data.empty:
                return float(price_data.iloc[0, 0])
            return None

        except Exception:
            return None

    def get_current_holdings(self) -> Dict[str, Position]:
        """获取当前持仓"""
        return {stock_id: pos for stock_id, pos in self.positions.items() if pos.amount > 0}

    def get_total_value(self) -> float:
        """获取总资产价值"""
        return self.total_value

    def get_account_info(self) -> Dict[str, Any]:
        """获取账户信息"""
        current_positions = self.get_current_holdings()

        position_info = {
            stock_id: pos.get_position_info()
            for stock_id, pos in current_positions.items()
        }

        return {
            "init_cash": self.init_cash,
            "current_cash": self.current_cash,
            "total_value": self.total_value,
            "position_count": len(current_positions),
            "position_details": position_info,
            "total_return": (self.total_value / self.init_cash - 1) if self.init_cash > 0 else 0,
            "cash_ratio": self.current_cash / self.total_value if self.total_value > 0 else 0,
            "position_value": self.total_value - self.current_cash
        }

    def get_daily_returns(self) -> pd.Series:
        """获取每日收益率"""
        if not self.daily_values:
            return pd.Series()

        df = pd.DataFrame(self.daily_values)
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)

        returns = df["total_value"].pct_change().fillna(0)
        return returns

    def get_performance_metrics(self) -> Dict[str, float]:
        """获取绩效指标"""
        returns = self.get_daily_returns()

        if returns.empty:
            return {}

        # 基础指标
        total_return = (self.total_value / self.init_cash - 1) if self.init_cash > 0 else 0
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0

        # 回撤指标
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
        }
```

## 实际应用示例

### 完整的回测流程

```python
import qlib
from qlib.data import D
from qlib.backtest.executor import SimulatorExecutor
from qlib.backtest.strategy import BaseStrategy
from qlib.backtest.decision import Order, BaseTradeDecision
from qlib.backtest.exchange import Exchange
from qlib.backtest.account import Account
import pandas as pd
import numpy as np

class SimpleMomentumStrategy(BaseStrategy):
    """
    简单动量策略示例

    策略逻辑：
    1. 计算过去20日的动量因子
    2. 选择动量最高的前10只股票
    3. 等权重配置
    4. 每月调仓
    """

    def __init__(self, top_k=10, rebalance_freq=20, **kwargs):
        """
        初始化策略

        Args:
            top_k: 选择股票数量
            rebalance_freq: 调仓频率（交易日）
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.top_k = top_k
        self.rebalance_freq = rebalance_freq
        self.last_rebalance_date = None
        self.current_holdings = {}

    def generate_trade_decision(self, trade_step, account, exchange, **kwargs):
        """
        生成交易决策

        Args:
            trade_step: 当前交易日期
            account: 账户信息
            exchange: 交易所信息
            **kwargs: 其他参数

        Returns:
            交易决策对象
        """
        # 1. 检查是否需要调仓
        if self._should_rebalance(trade_step):
            return self._rebalance_portfolio(trade_step, account, exchange)
        else:
            return self._hold_current_position()

    def _should_rebalance(self, trade_step) -> bool:
        """判断是否需要调仓"""
        if self.last_rebalance_date is None:
            return True

        days_since_last = (trade_step - self.last_rebalance_date).days
        return days_since_last >= self.rebalance_freq

    def _rebalance_portfolio(self, trade_step, account, exchange, **kwargs):
        """调仓操作"""
        # 1. 获取股票池
        stock_pool = D.instruments(market="csi300")

        # 2. 计算动量因子
        momentum_scores = self._calculate_momentum(stock_pool, trade_step)

        # 3. 选择top-k股票
        top_stocks = momentum_scores.nlargest(self.top_k).index.tolist()

        # 4. 生成订单
        current_value = account.get_total_value()
        target_value_per_stock = current_value / self.top_k

        orders = []
        current_holdings = account.get_current_holdings()

        # 4.1 卖出不在目标列表中的持仓
        for stock_id, position in current_holdings.items():
            if stock_id not in top_stocks:
                current_price = self._get_current_price(stock_id, trade_step)
                if current_price:
                    sell_amount = position.amount
                    orders.append(Order(
                        stock_id=stock_id,
                        amount=sell_amount,
                        direction=-1,  # 卖出
                        price=current_price
                    ))

        # 4.2 买入目标股票
        for stock_id in top_stocks:
            current_price = self._get_current_price(stock_id, trade_step)
            if current_price:
                target_amount = int(target_value_per_stock / current_price / 100) * 100  # 整手

                if stock_id in current_holdings:
                    current_amount = current_holdings[stock_id].amount
                    if target_amount > current_amount:
                        buy_amount = target_amount - current_amount
                        if buy_amount > 0:
                            orders.append(Order(
                                stock_id=stock_id,
                                amount=buy_amount,
                                direction=1,  # 买入
                                price=current_price
                            ))
                else:
                    if target_amount > 0:
                        orders.append(Order(
                            stock_id=stock_id,
                            amount=target_amount,
                            direction=1,  # 买入
                            price=current_price
                        ))

        # 5. 更新状态
        self.last_rebalance_date = trade_step
        self.current_holdings = {stock: target_value_per_stock for stock in top_stocks}

        # 6. 返回交易决策
        from qlib.backtest.decision import TradeDecisionOW
        return TradeDecisionOW(orders)

    def _hold_current_position(self):
        """保持当前持仓"""
        from qlib.backtest.decision import EmptyTradeDecision
        return EmptyTradeDecision()

    def _calculate_momentum(self, stock_pool, trade_step):
        """计算动量因子"""
        # 获取过去20日的价格数据
        end_date = trade_step.strftime("%Y-%m-%d")
        start_date = (trade_step - pd.Timedelta(days=30)).strftime("%Y-%m-%d")

        price_data = D.features(
            stock_pool,
            ['$close'],
            start_time=start_date,
            end_time=end_date,
            freq="day"
        )

        if price_data.empty:
            return pd.Series()

        # 计算20日动量
        momentum = price_data.groupby(level=1)['$close'].apply(
            lambda x: (x.iloc[-1] / x.iloc[0] - 1) if len(x) >= 20 else 0
        )

        return momentum

    def _get_current_price(self, stock_id, trade_step):
        """获取当前价格"""
        try:
            price_data = D.features(
                [stock_id],
                ['$close'],
                start_time=trade_step.strftime("%Y-%m-%d"),
                end_time=trade_step.strftime("%Y-%m-%d"),
                freq="day"
            )

            if not price_data.empty:
                return float(price_data.iloc[0, 0])
            return None
        except Exception:
            return None

def run_backtest():
    """运行回测"""
    # 1. 初始化Qlib
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")

    # 2. 设置回测参数
    instruments = D.instruments(market="csi300")
    start_time = "2020-01-01"
    end_time = "2023-12-31"

    # 3. 创建策略
    strategy = SimpleMomentumStrategy(top_k=10, rebalance_freq=20)

    # 4. 创建交易所
    exchange_kwargs = {
        "commission_rate": 0.0003,
        "tax_rate": 0.001,
        "min_commission": 5.0,
        "slippage_rate": 0.001,
        "market_impact_rate": 0.0001
    }
    exchange = Exchange(**exchange_kwargs)

    # 5. 创建账户
    account_kwargs = {
        "init_cash": 1e8  # 1亿初始资金
    }
    account = Account(**account_kwargs)

    # 6. 创建执行器
    executor = SimulatorExecutor(
        time_per_step="day",
        verbose=True,
        trade_exchange=exchange,
        trade_account=account
    )

    # 7. 执行回测
    backtest_result = executor.execute(
        strategy=strategy,
        start_time=start_time,
        end_time=end_time,
        trade_calendar=D.calendar()
    )

    # 8. 分析结果
    analyze_backtest_result(backtest_result)

    return backtest_result

def analyze_backtest_result(result):
    """分析回测结果"""
    if "error" in result:
        print(f"回测失败: {result['error']}")
        return

    # 1. 基础统计
    portfolio_analysis = result.get("portfolio_analysis", {})
    print("回测结果分析:")
    print(f"总收益率: {portfolio_analysis.get('total_return', 0):.2%}")
    print(f"夏普比率: {portfolio_analysis.get('sharpe_ratio', 0):.4f}")
    print(f"最大回撤: {portfolio_analysis.get('max_drawdown', 0):.2%}")

    # 2. 交易统计
    trade_stats = result.get("trade_statistics", {})
    print(f"\n交易统计:")
    print(f"总交易次数: {trade_stats.get('total_trades', 0)}")
    print(f"盈利交易次数: {trade_stats.get('profitable_trades', 0)}")
    print(f"总交易成本: {trade_stats.get('total_execution_cost', 0):,.2f}")

    # 3. 绘制收益曲线
    if "return_curve" in result:
        return_curve = result["return_curve"]
        print(f"\n收益曲线:")
        print(f"起始日期: {return_curve.index[0]}")
        print(f"结束日期: {return_curve.index[-1]}")
        print(f"最高点收益率: {return_curve.max():.2%}")
        print(f"最低点收益率: {return_curve.min():.2%}")

    # 4. 持仓分析
    trade_records = result.get("trade_records")
    if trade_records is not None and not trade_records.empty:
        print(f"\n持仓分析:")
        print(f"平均资产价值: {trade_records['portfolio_value'].mean():,.2f}")
        print(f"资产价值波动率: {trade_records['portfolio_value'].std():,.2f}")

if __name__ == "__main__":
    result = run_backtest()
```

## 总结

Qlib回测系统通过以下核心设计实现了专业级的量化回测能力：

### 技术特性

1. **事件驱动架构**: 基于时间的回测流程，模拟真实交易环境
2. **模块化设计**: 清晰的组件分离，便于扩展和定制
3. **真实成本建模**: 包含手续费、印花税、滑点等真实交易成本
4. **风险控制**: 涨跌停限制、市场冲击等风险管理机制
5. **详细分析**: 完整的回测报告和绩效分析

### 设计优势

1. **准确性**: 真实的市场环境模拟和成本建模
2. **灵活性**: 支持多种策略类型和执行模式
3. **可扩展性**: 模块化架构便于功能扩展
4. **实用性**: 专门针对量化投资场景优化
5. **可靠性**: 完善的错误处理和边界条件处理

### 最佳实践

1. **参数调优**: 合理设置交易成本和滑点参数
2. **样本外测试**: 严格的样本内/外测试分离
3. **风险管理**: 重视回撤控制和风险限制
4. **成本分析**: 详细分析交易成本对策略的影响
5. **结果验证**: 多角度验证回测结果的可靠性

Qlib的回测系统为量化投资研究提供了专业而可靠的验证平台，使研究者能够准确评估策略的有效性，为实盘交易提供科学依据。通过深入理解这些核心技术，量化研究者可以构建更加准确和可靠的回测系统。