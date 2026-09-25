"""Interactive Brokers adapter built on ib_async (maintained successor of ib_insync).

Requires a running TWS or IB Gateway with the API enabled.
"""

from __future__ import annotations

import logging
import math
import time

from ib_async import IB, LimitOrder, Stock, Ticker, Trade

from ..account import AccountInfo
from ..config import IBConfig
from ..models import Fill, OrderIntent, Side, Tick
from .base import DoneCallback, FillCallback, TickCallback

log = logging.getLogger(__name__)


OTC_EXCHANGES = frozenset({"PINK", "OTC", "OTCBB", "OTCQB", "OTCQX", "GREY", "EXPERT"})


def _num(x: float | None) -> float:
    return 0.0 if x is None or math.isnan(x) else float(x)


class IBKRBroker:
    def __init__(self, config: IBConfig):
        self.config = config
        self.ib = IB()
        self._contracts: dict[str, Stock] = {}
        self._trades: dict[str, Trade] = {}

    async def connect(self) -> None:
        c = self.config
        await self.ib.connectAsync(c.host, c.port, clientId=c.client_id)
        self.ib.errorEvent += self._on_error
        log.info("connected to IBKR %s:%s (client %s)", c.host, c.port, c.client_id)

    def contract(self, symbol: str) -> Stock | None:
        """Qualified contract for a subscribed symbol (used by the IBKR news source)."""
        return self._contracts.get(symbol)

    async def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()

    async def subscribe(self, symbols: list[str], on_tick: TickCallback) -> None:
        contracts = [Stock(s, "SMART", "USD") for s in symbols]
        await self.ib.qualifyContractsAsync(*contracts)
        by_con_id: dict[int, str] = {}
        for sym, contract in zip(symbols, contracts, strict=True):
            if not contract.conId:
                log.warning("%s: not found at IBKR; skipping", sym)
                continue
            if contract.primaryExchange.upper() in OTC_EXCHANGES:
                # IBKR restricts buying many OTC stocks, and their data/news are poor.
                log.warning("%s trades OTC (%s); skipping", sym, contract.primaryExchange)
                continue
            self._contracts[sym] = contract
            by_con_id[contract.conId] = sym
            self.ib.reqMktData(contract, "", False, False)

        def handle(tickers: set[Ticker]) -> None:
            for t in tickers:
                sym = by_con_id.get(t.contract.conId)
                if sym:
                    on_tick(
                        Tick(
                            sym,
                            _num(t.bid),
                            _num(t.ask),
                            _num(t.last),
                            time.time(),
                            _num(t.bidSize),
                            _num(t.askSize),
                        )
                    )

        self.ib.pendingTickersEvent += handle

    def place_limit(self, intent: OrderIntent, on_fill: FillCallback, on_done: DoneCallback) -> str:
        contract = self._contracts[intent.symbol]
        order = LimitOrder(intent.side.value, intent.qty, intent.limit_price, tif="DAY")
        trade = self.ib.placeOrder(contract, order)
        oid = str(trade.order.orderId)
        self._trades[oid] = trade
        done = False

        def fill_handler(_trade: Trade, fill) -> None:
            ex = fill.execution
            on_fill(
                Fill(
                    oid,
                    intent.symbol,
                    Side(intent.side.value),
                    int(ex.shares),
                    float(ex.price),
                    time.time(),
                )
            )

        def status_handler(t: Trade) -> None:
            nonlocal done
            if not done and t.isDone():
                done = True
                self._trades.pop(oid, None)
                on_done(oid, int(t.remaining()))

        def commission_handler(_trade: Trade, _fill, report) -> None:
            fee = _num(report.commission)
            if fee:
                on_fill(Fill(oid, intent.symbol, intent.side, 0, 0.0, time.time(), fee))

        trade.fillEvent += fill_handler
        trade.statusEvent += status_handler
        trade.commissionReportEvent += commission_handler
        return oid

    def cancel(self, order_id: str) -> None:
        trade = self._trades.get(order_id)
        if trade and not trade.isDone():
            self.ib.cancelOrder(trade.order)

    def cancel_all(self) -> None:
        # Only this client's orders -- also catches ones left over from a crashed run, which
        # IBKR re-reports on connect. reqGlobalCancel would kill manually placed orders too.
        for trade in self.ib.openTrades():
            if trade.order.clientId == self.config.client_id and not trade.isDone():
                self.ib.cancelOrder(trade.order)

    async def account(self) -> AccountInfo:
        values: dict[str, str] = {}
        for v in self.ib.accountValues():
            if v.currency in ("USD", "BASE", ""):
                values.setdefault(v.tag, v.value)

        def num(tag: str) -> float | None:
            try:
                return float(values[tag])
            except (KeyError, ValueError):
                return None

        remaining = num("DayTradesRemaining")
        return AccountInfo(
            equity=num("NetLiquidation"),
            settled_cash=num("SettledCash"),
            day_trades_remaining=None if remaining is None else int(remaining),
        )

    async def positions(self) -> dict[str, tuple[int, float]]:
        out: dict[str, tuple[int, float]] = {}
        for pos in self.ib.positions():
            if pos.contract.secType != "STK":
                continue
            qty, avg = out.get(pos.contract.symbol, (0, 0.0))
            new_qty = qty + int(pos.position)
            # avgCost for stocks is per share (including commissions).
            new_avg = (avg * qty + pos.avgCost * pos.position) / new_qty if new_qty else 0.0
            out[pos.contract.symbol] = (new_qty, new_avg)
        return out

    def _on_error(self, req_id: int, code: int, message: str, *_: object) -> None:
        log.warning("IBKR error reqId=%s code=%s: %s", req_id, code, message)
