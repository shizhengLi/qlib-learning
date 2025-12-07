# Qlib数据采集器深度解析：构建可靠的金融数据管道

## 引言

数据是量化投资的基础，高质量的数据管道是成功策略的前提。Qlib提供了强大的数据采集系统，支持从多个数据源获取、处理和存储金融数据。本文将深入分析Qlib数据采集器的设计架构和实现细节，帮助读者构建可靠的数据基础设施。

## 数据采集器架构概览

Qlib数据采集器采用分层的架构设计，包含数据源适配、数据清洗、格式转换和存储管理等核心组件。

## 核心数据源

### 1. Yahoo Finance数据采集器

Yahoo Finance是免费的历史股价数据源，适合研究和学习使用：

```python
import yfinance as yf
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

class YahooFinanceCollector:
    """Yahoo Finance数据采集器"""

    def __init__(self, cache_dir: str = "./data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def collect_stock_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        采集股票数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            股票数据字典 {symbol: DataFrame}
        """
        data_dict = {}

        for symbol in symbols:
            try:
                print(f"Collecting data for {symbol}...")

                # 使用yfinance下载数据
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)

                if not df.empty:
                    # 标准化列名
                    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
                    df.index.name = 'date'

                    # 数据验证
                    df = self._validate_data(df)

                    data_dict[symbol] = df

            except Exception as e:
                print(f"Error collecting {symbol}: {e}")
                continue

        return data_dict

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据验证和清洗"""
        # 1. 检查必要列
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # 2. 数据类型转换
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 3. 异常值处理
        df = self._handle_outliers(df)

        # 4. 缺失值处理
        df = self._handle_missing_values(df)

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """异常值处理"""
        price_cols = ['open', 'high', 'low', 'close']

        for col in price_cols:
            # 价格不能为负数
            df[col] = df[col].clip(lower=0)

            # 处理极端价格变化
            returns = df[col].pct_change()
            extreme_change = (returns.abs() > 0.5)  # 单日涨跌幅超过50%
            if extreme_change.any():
                print(f"Warning: Found extreme price changes in {col}")
                # 用前值填充异常值
                df.loc[extreme_change, col] = df[col].fillna(method='ffill')

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """缺失值处理"""
        # 前向填充价格数据
        price_cols = ['open', 'high', 'low', 'close']
        df[price_cols] = df[price_cols].fillna(method='ffill').fillna(method='bfill')

        # 成交量缺失填充为0
        df['volume'] = df['volume'].fillna(0)

        return df

    def save_to_qlib_format(self, data_dict: Dict[str, pd.DataFrame], output_dir: str):
        """保存为Qlib格式"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 创建目录结构
        (output_path / "instruments").mkdir(exist_ok=True)
        (output_path / "features" / "day").mkdir(parents=True, exist_ok=True)

        # 保存股票列表
        symbols = list(data_dict.keys())
        instruments_df = pd.DataFrame({'instrument': symbols})
        instruments_df.to_csv(output_path / "instruments" / "all.csv", index=False)

        # 保存价格数据
        for symbol, df in data_dict.items():
            # 重采样到日频（如果需要）
            df_daily = df.resample('D').last().dropna()

            # 保存为CSV格式
            file_path = output_path / "features" / "day" / f"{symbol}.csv"
            df_daily.to_csv(file_path)

# 使用示例
def collect_yahoo_data():
    """采集Yahoo Finance数据示例"""
    collector = YahooFinanceCollector()

    # 沪深300成分股（简化示例）
    symbols = [
        '000001.SZ', '000002.SZ', '000858.SZ', '002415.SZ', '002594.SZ',
        '600000.SH', '600036.SH', '600519.SH', '600887.SH', '601318.SH'
    ]

    start_date = "2020-01-01"
    end_date = "2023-12-31"

    # 采集数据
    data = collector.collect_stock_data(symbols, start_date, end_date)

    # 保存为Qlib格式
    collector.save_to_qlib_format(data, "./qlib_data/cn_data")

    print(f"Successfully collected data for {len(data)} stocks")
```

### 2. Tushare数据采集器

Tushare提供了更全面的A股数据，包括基本面、技术指标和资金流向等：

```python
import tushare as ts
import pandas as pd
from typing import List, Dict, Optional
import time

class TushareCollector:
    """Tushare数据采集器"""

    def __init__(self, token: str):
        """
        初始化Tushare采集器

        Args:
            token: Tushare API token
        """
        ts.set_token(token)
        self.pro = ts.pro_api()

    def get_stock_basic(self) -> pd.DataFrame:
        """获取股票基本信息"""
        # 获取股票列表
        stock_basic = self.pro.stock_basic(
            exchange='',
            list_status='L',
            fields='ts_code,symbol,name,area,industry,list_date'
        )
        return stock_basic

    def get_daily_data(self, ts_codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """获取日线行情数据"""
        all_data = []

        for ts_code in ts_codes:
            try:
                print(f"Fetching daily data for {ts_code}")

                # 获取日线数据
                df_daily = self.pro.daily(
                    ts_code=ts_code,
                    start_date=start_date.replace('-', ''),
                    end_date=end_date.replace('- '')
                )

                if not df_daily.empty:
                    # 添加前复权数据
                    df_qfq = self.pro.daily_basic(
                        ts_code=ts_code,
                        start_date=start_date.replace('-', ''),
                        end_date=end_date.replace('-', ''),
                        fields='ts_code,trade_date,close,turnoverf_ratio,volume_ratio,pe,pb'
                    )

                    # 合并数据
                    df_merged = pd.merge(df_daily, df_qfq, on=['ts_code', 'trade_date'], how='left')
                    all_data.append(df_merged)

                # API限频控制
                time.sleep(0.1)

            except Exception as e:
                print(f"Error fetching data for {ts_code}: {e}")
                continue

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()

    def get_financial_data(self, ts_codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """获取财务数据"""
        all_financial = []

        for ts_code in ts_codes:
            try:
                print(f"Fetching financial data for {ts_code}")

                # 获取财务指标
                df_fina = self.pro.fina_indicator(
                    ts_code=ts_code,
                    start_date=start_date.replace('-', ''),
                    end_date=end_date.replace('-', '')
                )

                if not df_fina.empty:
                    all_financial.append(df_fina)

                # API限频控制
                time.sleep(0.1)

            except Exception as e:
                print(f"Error fetching financial data for {ts_code}: {e}")
                continue

        if all_financial:
            return pd.concat(all_financial, ignore_index=True)
        else:
            return pd.DataFrame()

    def get_money_flow(self, ts_codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """获取资金流向数据"""
        all_flow = []

        for ts_code in ts_codes:
            try:
                print(f"Fetching money flow data for {ts_code}")

                df_moneyflow = self.pro.moneyflow(
                    ts_code=ts_code,
                    start_date=start_date.replace('-', ''),
                    end_date=end_date.replace('-', '')
                )

                if not df_moneyflow.empty:
                    all_flow.append(df_moneyflow)

                # API限频控制
                time.sleep(0.1)

            except Exception as e:
                print(f"Error fetching money flow data for {ts_code}: {e}")
                continue

        if all_flow:
            return pd.concat(all_flow, ignore_index=True)
        else:
            return pd.DataFrame()

    def process_tushare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理Tushare数据格式"""
        # 1. 日期格式转换
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)

        # 2. 列名标准化
        column_mapping = {
            'ts_code': 'symbol',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'vol': 'volume',
            'amount': 'amount'
        }

        df.rename(columns=column_mapping, inplace=True)

        # 3. 数据类型转换
        numeric_cols = ['open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

# 使用示例
def collect_tushare_data():
    """采集Tushare数据示例"""
    # 需要申请Tushare token
    token = "your_tushare_token_here"

    collector = TushareCollector(token)

    # 1. 获取股票列表
    stock_basic = collector.get_stock_basic()
    print(f"Found {len(stock_basic)} stocks")

    # 2. 选择部分股票进行示例
    ts_codes = stock_basic['ts_code'].head(50).tolist()

    start_date = "2020-01-01"
    end_date = "2023-12-31"

    # 3. 采集不同类型的数据
    daily_data = collector.get_daily_data(ts_codes, start_date, end_date)
    financial_data = collector.get_financial_data(ts_codes, start_date, end_date)
    moneyflow_data = collector.get_money_flow(ts_codes, start_date, end_date)

    print(f"Daily data shape: {daily_data.shape}")
    print(f"Financial data shape: {financial_data.shape}")
    print(f"Money flow data shape: {moneyflow_data.shape}")

    return {
        'daily': daily_data,
        'financial': financial_data,
        'moneyflow': moneyflow_data,
        'stock_basic': stock_basic
    }
```

### 3. US股票数据采集器

支持美股数据的采集和处理：

```python
import requests
import pandas as pd
from typing import List, Dict
import time

class USStockCollector:
    """美股数据采集器"""

    def __init__(self, api_key: str = None):
        """
        初始化美股采集器

        Args:
            api_key: API密钥（可选）
        """
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

    def get_sp500_symbols(self) -> List[str]:
        """获取S&P 500成分股列表"""
        # 这里使用硬编码的示例列表，实际应用中可以从维基百科等来源获取
        sp500_symbols = [
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'JPM',
            'JNJ', 'V', 'PG', 'HD', 'UNH', 'MA', 'BAC', 'XOM', 'CVX',
            'LLY', 'ABBV', 'PFE', 'KO', 'PEP', 'TMO', 'COST', 'AVGO',
            'LIN', 'CRM', 'ACN', 'NKE', 'DHR', 'WFC', 'TXN', 'NEE'
        ]
        return sp500_symbols

    def collect_daily_prices(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """采集日线价格数据"""
        data_dict = {}

        for symbol in symbols:
            try:
                print(f"Collecting data for {symbol}...")

                if self.api_key:
                    # 使用Alpha Vantage API
                    df = self._get_alpha_vantage_data(symbol, self.api_key)
                else:
                    # 使用yfinance作为备选
                    df = self._get_yahoo_data(symbol, start_date, end_date)

                if df is not None and not df.empty:
                    data_dict[symbol] = df

                # API限频
                if self.api_key:
                    time.sleep(12)  # Alpha Vantage免费版限制

            except Exception as e:
                print(f"Error collecting {symbol}: {e}")
                continue

        return data_dict

    def _get_alpha_vantage_data(self, symbol: str, api_key: str) -> pd.DataFrame:
        """使用Alpha Vantage API获取数据"""
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': 'full',
            'apikey': api_key
        }

        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()

            if 'Time Series (Daily)' in data:
                # 转换为DataFrame
                df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
                df.index = pd.to_datetime(df.index)

                # 重命名列
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                df = df.astype(float)

                # 按日期排序
                df = df.sort_index()

                return df
            else:
                print(f"No data returned for {symbol}")
                return None

        except Exception as e:
            print(f"Error fetching Alpha Vantage data for {symbol}: {e}")
            return None

    def _get_yahoo_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """使用yfinance获取数据"""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)

            if not df.empty:
                # 标准化列名
                df.columns = [col.lower().replace(' ', '_') for col in df.columns]
                df.index.name = 'date'

                # 选择需要的列
                df = df[['open', 'high', 'low', 'close', 'volume']]

                return df

        except Exception as e:
            print(f"Error fetching Yahoo data for {symbol}: {e}")

        return None

    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算收益率"""
        # 日收益率
        df['return'] = df['close'].pct_change()

        # 对数收益率
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))

        # 累计收益率
        df['cum_return'] = (1 + df['return']).cumprod() - 1

        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标"""
        # 移动平均线
        for period in [5, 10, 20, 50, 200]:
            df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'close_ma_{period}_ratio'] = df['close'] / df[f'ma_{period}']

        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])

        # 布林带
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])

        # ATR
        df['atr'] = self._calculate_atr(df)

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2):
        """计算布林带"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算ATR（平均真实范围）"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr

# 使用示例
def collect_us_stock_data():
    """采集美股数据示例"""
    collector = USStockCollector()

    # 获取S&P 500成分股
    symbols = collector.get_sp500_symbols()

    start_date = "2020-01-01"
    end_date = "2023-12-31"

    # 采集数据（只采集前10只作为示例）
    sample_symbols = symbols[:10]
    data = collector.collect_daily_prices(sample_symbols, start_date, end_date)

    # 处理数据
    processed_data = {}
    for symbol, df in data.items():
        # 计算收益率
        df = collector.calculate_returns(df)

        # 添加技术指标
        df = collector.add_technical_indicators(df)

        processed_data[symbol] = df

    print(f"Successfully collected data for {len(processed_data)} stocks")
    return processed_data
```

## 数据质量控制

### 数据验证框架

```python
from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np

class DataValidator:
    """数据质量验证器"""

    def __init__(self):
        self.validation_rules = []

    def add_rule(self, rule_func, description: str, severity: str = "error"):
        """
        添加验证规则

        Args:
            rule_func: 验证函数，返回(is_valid, message)
            description: 规则描述
            severity: 严重程度 ("error", "warning", "info")
        """
        self.validation_rules.append({
            'func': rule_func,
            'description': description,
            'severity': severity
        })

    def validate_dataframe(self, df: pd.DataFrame, data_name: str = "data") -> Tuple[bool, List[Dict]]:
        """
        验证DataFrame

        Returns:
            (is_valid, issues_list)
        """
        issues = []

        for rule in self.validation_rules:
            try:
                is_valid, message = rule['func'](df)
                if not is_valid:
                    issues.append({
                        'rule': rule['description'],
                        'message': message,
                        'severity': rule['severity'],
                        'data': data_name
                    })
            except Exception as e:
                issues.append({
                    'rule': rule['description'],
                    'message': f"Validation error: {str(e)}",
                    'severity': 'error',
                    'data': data_name
                })

        # 检查是否有错误级别的问题
        has_errors = any(issue['severity'] == 'error' for issue in issues)

        return not has_errors, issues

# 预定义验证规则
def create_price_data_validator() -> DataValidator:
    """创建价格数据验证器"""
    validator = DataValidator()

    # 1. 基本列检查
    def check_required_columns(df):
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
        return True, "All required columns present"

    validator.add_rule(check_required_columns, "Required columns check")

    # 2. 价格合理性检查
    def check_price_relationships(df):
        # high >= low
        invalid_high_low = (df['high'] < df['low']).any()
        if invalid_high_low:
            return False, "High price is lower than low price in some rows"

        # close应该在high和low之间
        invalid_close = ((df['close'] > df['high']) | (df['close'] < df['low'])).any()
        if invalid_close:
            return False, "Close price is outside high-low range in some rows"

        return True, "Price relationships are valid"

    validator.add_rule(check_price_relationships, "Price relationship check")

    # 3. 数据类型检查
    def check_data_types(df):
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                return False, f"Column {col} is not numeric"
        return True, "All columns have correct data types"

    validator.add_rule(check_data_types, "Data type check")

    # 4. 缺失值检查
    def check_missing_values(df):
        missing_summary = df.isnull().sum()
        if missing_summary.sum() > 0:
            return False, f"Found missing values: {missing_summary[missing_summary > 0].to_dict()}"
        return True, "No missing values found"

    validator.add_rule(check_missing_values, "Missing values check")

    # 5. 异常值检查
    def check_outliers(df):
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                # 检查负价格
                negative_prices = (df[col] < 0).any()
                if negative_prices:
                    return False, f"Found negative prices in {col}"

                # 检查极端价格变化
                returns = df[col].pct_change()
                extreme_changes = (returns.abs() > 0.5).sum()
                if extreme_changes > 0:
                    return False, f"Found {extreme_changes} extreme price changes in {col}"

        return True, "No obvious outliers found"

    validator.add_rule(check_outliers, "Outlier check")

    return validator

# 使用示例
def validate_stock_data(data_dict: Dict[str, pd.DataFrame]):
    """验证股票数据质量"""
    validator = create_price_data_validator()

    validation_results = {}
    overall_valid = True

    for symbol, df in data_dict.items():
        is_valid, issues = validator.validate_dataframe(df, symbol)
        validation_results[symbol] = {
            'is_valid': is_valid,
            'issues': issues
        }

        if not is_valid:
            overall_valid = False

        # 打印问题
        if issues:
            print(f"\nIssues found in {symbol}:")
            for issue in issues:
                print(f"  [{issue['severity'].upper()}] {issue['rule']}: {issue['message']}")

    if overall_valid:
        print("\n✅ All data validation checks passed!")
    else:
        print("\n❌ Data validation failed for some stocks")

    return validation_results
```

## 总结

Qlib数据采集器通过以下核心特性构建了可靠的数据管道：

### 技术特性

1. **多数据源支持**: Yahoo Finance、Tushare、Alpha Vantage等
2. **数据质量保证**: 完善的验证和清洗机制
3. **标准化处理**: 统一的数据格式和接口
4. **错误处理**: 健壮的异常处理和重试机制
5. **性能优化**: 缓存和批量处理优化

### 实践建议

1. **数据源选择**: 根据需求选择合适的数据源
2. **质量控制**: 重视数据验证和清洗
3. **存储策略**: 合理的数据存储和索引
4. **更新机制**: 定期数据更新和维护
5. **监控告警**: 数据质量监控和异常告警

通过构建可靠的数据采集和管道系统，为量化研究提供高质量的数据基础。