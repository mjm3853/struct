"""LangGraph tool definitions wrapping yfinance market data."""

from langchain_core.tools import tool

from struct_agent import client


@tool
def get_stock_quote(ticker: str) -> str:
    """Get current quote data for a stock including price, volume, and key metrics.

    Returns the current price, day range, volume, market cap, P/E ratio,
    and 52-week range. Use this as a starting point to understand where
    a stock is trading right now.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL, TSLA, MSFT).
    """
    q = client.get_quote(ticker)
    change = q.price - q.previous_close
    change_pct = (change / q.previous_close * 100) if q.previous_close else 0
    mcap = f"${q.market_cap / 1e9:.1f}B" if q.market_cap else "N/A"
    pe = f"{q.pe_ratio:.1f}" if q.pe_ratio else "N/A"
    return (
        f"{q.name} ({q.symbol})\n"
        f"  Price: ${q.price:.2f} ({change:+.2f}, {change_pct:+.1f}%)\n"
        f"  Day range: ${q.day_low:.2f} - ${q.day_high:.2f}\n"
        f"  Volume: {q.volume:,}\n"
        f"  Market cap: {mcap}  |  P/E: {pe}\n"
        f"  52-wk range: ${q.fifty_two_week_low:.2f} - ${q.fifty_two_week_high:.2f}"
    )


@tool
def get_option_chain(
    ticker: str,
    expiration: str | None = None,
    option_type: str = "both",
    min_volume: int = 0,
    max_results: int = 20,
) -> str:
    """Get the options chain for a stock at a specific expiration date.

    Returns strike, bid, ask, volume, open interest, and implied volatility
    for call and/or put contracts. High volume relative to open interest
    signals fresh positioning. Elevated IV indicates the market expects
    larger moves.

    If no expiration is provided, automatically uses the nearest-term
    expiration and also lists other available dates.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL, TSLA).
        expiration: Expiration date as YYYY-MM-DD. If omitted, uses nearest expiration.
        option_type: "calls", "puts", or "both".
        min_volume: Only show contracts with at least this many contracts traded.
        max_results: Max contracts to return per side (calls/puts).
    """
    exps = client.get_option_expirations(ticker)
    if not exps:
        return f"No options data available for {ticker}."
    if not expiration:
        expiration = exps[0]

    chain = client.get_option_chain(ticker, expiration)
    lines = [f"Options chain for {chain.symbol} exp {chain.expiration}:", ""]

    def fmt_side(contracts, label):
        filtered = [c for c in contracts if (c.volume or 0) >= min_volume]
        filtered.sort(key=lambda c: c.volume or 0, reverse=True)
        filtered = filtered[:max_results]
        if not filtered:
            lines.append(f"  No {label} matching filters.")
            return
        lines.append(f"  {label}:")
        for c in filtered:
            itm = " ITM" if c.in_the_money else ""
            vol = f"{c.volume:,}" if c.volume else "0"
            lines.append(
                f"    ${c.strike:.0f}: bid ${c.bid:.2f} / ask ${c.ask:.2f}, "
                f"vol {vol}, OI {c.open_interest:,}, "
                f"IV {c.implied_volatility:.1%}{itm}"
            )

    if option_type in ("both", "calls"):
        fmt_side(chain.calls, "CALLS")
    if option_type in ("both", "puts"):
        fmt_side(chain.puts, "PUTS")

    other_exps = [e for e in exps if e != expiration][:10]
    if other_exps:
        lines.append(f"\n  Other expirations: {', '.join(other_exps)}")

    return "\n".join(lines)


@tool
def get_price_history(
    ticker: str,
    period: str = "1mo",
    interval: str = "1d",
) -> str:
    """Get historical OHLCV price data for a stock.

    Useful for understanding recent price action, trends, and volume patterns.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL, TSLA).
        period: Time period — 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
        interval: Bar interval — 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo.
    """
    bars = client.get_history(ticker, period=period, interval=interval)
    if not bars:
        return f"No price history for {ticker} with period={period}."
    lines = [f"Price history for {ticker} ({period}, {interval}):", ""]
    for b in bars:
        lines.append(f"  {b.date}: O ${b.open} H ${b.high} L ${b.low} C ${b.close} V {b.volume:,}")
    return "\n".join(lines)


@tool
def get_institutional_holders(ticker: str) -> str:
    """Get top institutional holders of a stock.

    Shows the largest institutional shareholders, their share counts,
    percentage of outstanding shares, and most recent reporting date.
    Useful for understanding who the major holders are and whether
    institutions are accumulating or reducing positions.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL, TSLA).
    """
    holders = client.get_institutional_holders(ticker)
    if not holders:
        return f"No institutional holder data for {ticker}."
    lines = [f"Top institutional holders of {ticker}:", ""]
    for h in holders:
        lines.append(
            f"  {h.holder}: {h.shares:,} shares "
            f"({h.pct_held:.2%}), ${h.value:,} — reported {h.date_reported}"
        )
    return "\n".join(lines)


ALL_TOOLS = [
    get_stock_quote,
    get_option_chain,
    get_price_history,
    get_institutional_holders,
]
