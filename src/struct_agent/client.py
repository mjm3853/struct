"""Data client wrapping yfinance for structured financial market data."""

from dataclasses import dataclass
from typing import Any

import yfinance as yf


@dataclass
class TickerQuote:
    symbol: str
    price: float
    previous_close: float
    open: float
    day_high: float
    day_low: float
    volume: int
    market_cap: int | None
    pe_ratio: float | None
    fifty_two_week_high: float
    fifty_two_week_low: float
    name: str


@dataclass
class OptionContract:
    strike: float
    bid: float
    ask: float
    last_price: float
    volume: int | None
    open_interest: int
    implied_volatility: float
    in_the_money: bool


@dataclass
class OptionChain:
    symbol: str
    expiration: str
    calls: list[OptionContract]
    puts: list[OptionContract]


@dataclass
class OHLCVBar:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class InstitutionalHolder:
    holder: str
    shares: int
    date_reported: str
    pct_held: float
    value: int


def _safe(info: dict[str, Any], key: str, default=None):
    """Extract a value from yfinance info dict, handling KeyError and None."""
    v = info.get(key, default)
    return default if v is None else v


def get_quote(symbol: str) -> TickerQuote:
    t = yf.Ticker(symbol)
    info = t.info
    return TickerQuote(
        symbol=symbol.upper(),
        price=_safe(info, "currentPrice", _safe(info, "regularMarketPrice", 0.0)),
        previous_close=_safe(info, "previousClose", 0.0),
        open=_safe(info, "open", _safe(info, "regularMarketOpen", 0.0)),
        day_high=_safe(info, "dayHigh", _safe(info, "regularMarketDayHigh", 0.0)),
        day_low=_safe(info, "dayLow", _safe(info, "regularMarketDayLow", 0.0)),
        volume=_safe(info, "volume", _safe(info, "regularMarketVolume", 0)),
        market_cap=_safe(info, "marketCap"),
        pe_ratio=_safe(info, "trailingPE"),
        fifty_two_week_high=_safe(info, "fiftyTwoWeekHigh", 0.0),
        fifty_two_week_low=_safe(info, "fiftyTwoWeekLow", 0.0),
        name=_safe(info, "shortName", symbol),
    )


def get_option_expirations(symbol: str) -> list[str]:
    return list(yf.Ticker(symbol).options)


def get_option_chain(symbol: str, expiration: str) -> OptionChain:
    chain = yf.Ticker(symbol).option_chain(expiration)

    def parse_contracts(df) -> list[OptionContract]:
        contracts = []
        for _, row in df.iterrows():
            contracts.append(
                OptionContract(
                    strike=float(row["strike"]),
                    bid=float(row.get("bid", 0)),
                    ask=float(row.get("ask", 0)),
                    last_price=float(row.get("lastPrice", 0)),
                    volume=int(row["volume"]) if row.get("volume") else None,
                    open_interest=int(row.get("openInterest", 0)),
                    implied_volatility=float(row.get("impliedVolatility", 0)),
                    in_the_money=bool(row.get("inTheMoney", False)),
                )
            )
        return contracts

    return OptionChain(
        symbol=symbol.upper(),
        expiration=expiration,
        calls=parse_contracts(chain.calls),
        puts=parse_contracts(chain.puts),
    )


def get_history(symbol: str, period: str = "1mo", interval: str = "1d") -> list[OHLCVBar]:
    df = yf.Ticker(symbol).history(period=period, interval=interval)
    bars = []
    for date, row in df.iterrows():
        bars.append(
            OHLCVBar(
                date=str(date.date()) if hasattr(date, "date") else str(date),
                open=round(float(row["Open"]), 2),
                high=round(float(row["High"]), 2),
                low=round(float(row["Low"]), 2),
                close=round(float(row["Close"]), 2),
                volume=int(row["Volume"]),
            )
        )
    return bars


def get_institutional_holders(symbol: str) -> list[InstitutionalHolder]:
    df = yf.Ticker(symbol).institutional_holders
    if df is None or df.empty:
        return []
    holders = []
    for _, row in df.iterrows():
        holders.append(
            InstitutionalHolder(
                holder=str(row.get("Holder", "")),
                shares=int(row.get("Shares", 0)),
                date_reported=str(row.get("Date Reported", "")),
                pct_held=float(row.get("% Out", 0)),
                value=int(row.get("Value", 0)),
            )
        )
    return holders
