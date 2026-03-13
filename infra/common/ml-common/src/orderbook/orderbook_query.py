"""
Orderbook Query Layer - PostgreSQL Integration

Day 3.2: Python module for querying orderbook snapshots from PostgreSQL

This module provides simple interface for:
- Fetching orderbook snapshots by symbol/time range
- Loading data for dataset generation
- Querying historical orderbook state

Usage:
 from ml_common.orderbook.orderbook_query import OrderbookQuery

 query = OrderbookQuery(host='localhost', port=5432, user='postgres', password='postgres', database='ml-framework')

 # Get latest snapshots
 latest = query.get_latest_snapshots(symbols=['BTCUSDT', 'ETHUSDT'], limit=10)

 # Get snapshots for time range
 snapshots = query.get_snapshots(
 symbol='BTCUSDT',
 start_time=datetime.now - timedelta(hours=24),
 end_time=datetime.now
 )

 # Get hourly aggregated snapshots (for 168-hour state vector)
 hourly = query.get_hourly_snapshots(symbol='BTCUSDT', hours=168)
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


class OrderbookQuery:
 """
 PostgreSQL query layer for orderbook snapshots

 Day 3.2: Simple interface for reading orderbook data from TimescaleDB
 """

 def __init__(
 self,
 host: str = 'localhost',
 port: int = 5432,
 user: str = 'postgres',
 password: str = 'postgres',
 database: str = 'ml-framework',
 schema: str = 'public'
 ):
 """Initialize database connection"""
 self.conn_params = {
 'host': host,
 'port': port,
 'user': user,
 'password': password,
 'database': database,
 }
 self.schema = schema
 self.conn = None

 logger.info(f"OrderbookQuery initialized: {database}@{host}:{port}")

 def connect(self):
 """Establish database connection"""
 if self.conn is None or self.conn.closed:
 self.conn = psycopg2.connect(**self.conn_params)
 logger.debug("Database connection established")

 def close(self):
 """Close database connection"""
 if self.conn and not self.conn.closed:
 self.conn.close
 logger.debug("Database connection closed")

 def __enter__(self):
 """Context manager entry"""
 self.connect
 return self

 def __exit__(self, exc_type, exc_val, exc_tb):
 """Context manager exit"""
 self.close

 def get_latest_snapshots(
 self,
 symbols: Optional[List[str]] = None,
 limit: int = 10
 ) -> List[Dict[str, Any]]:
 """
 Get latest orderbook snapshots

 Args:
 symbols: List of symbols (default: all symbols)
 limit: Number of snapshots per symbol

 Returns:
 List of snapshot dictionaries
 """
 self.connect

 if symbols is None:
 symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']

 with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
 query = f"""
 SELECT
 id,
 symbol,
 exchange,
 timestamp,
 mid_price,
 best_bid,
 best_ask,
 spread_pct,
 spread_bps,
 imbalance_10,
 imbalance_20,
 bid_depth_10,
 ask_depth_10,
 depth,
 bids,
 asks,
 wall_detection,
 absorption_metrics,
 trade_metrics,
 created_at
 FROM {self.schema}.orderbooks
 WHERE symbol = ANY(%s)
 ORDER BY timestamp DESC
 LIMIT %s;
 """

 cur.execute(query, (symbols, limit * len(symbols)))
 rows = cur.fetchall

 logger.info(f"Fetched {len(rows)} latest snapshots for {len(symbols)} symbols")
 return [dict(row) for row in rows]

 def get_snapshots(
 self,
 symbol: str,
 start_time: datetime,
 end_time: datetime
 ) -> List[Dict[str, Any]]:
 """
 Get orderbook snapshots for time range

 Args:
 symbol: Trading symbol (e.g., 'BTCUSDT')
 start_time: Start timestamp
 end_time: End timestamp

 Returns:
 List of snapshot dictionaries ordered by timestamp ASC
 """
 self.connect

 with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
 query = f"""
 SELECT
 id,
 symbol,
 exchange,
 timestamp,
 mid_price,
 best_bid,
 best_ask,
 spread_pct,
 spread_bps,
 imbalance_10,
 imbalance_20,
 bid_depth_10,
 ask_depth_10,
 depth,
 bids,
 asks,
 wall_detection,
 absorption_metrics,
 trade_metrics
 FROM {self.schema}.orderbooks
 WHERE symbol = %s
 AND timestamp >= %s
 AND timestamp <= %s
 ORDER BY timestamp ASC;
 """

 cur.execute(query, (symbol, start_time, end_time))
 rows = cur.fetchall

 logger.info(f"Fetched {len(rows)} snapshots for {symbol} from {start_time} to {end_time}")
 return [dict(row) for row in rows]

 def get_hourly_snapshots(
 self,
 symbol: str,
 hours: int = 168
 ) -> List[Dict[str, Any]]:
 """
 Get hourly aggregated orderbook snapshots

 Args:
 symbol: Trading symbol
 hours: Number of hours to fetch (default: 168 = 7 days)

 Returns:
 List of 168 hourly snapshots (one per hour) ordered by timestamp ASC
 """
 self.connect

 end_time = datetime.now(timezone.utc)
 start_time = end_time - timedelta(hours=hours)

 with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
 # Get one snapshot per hour (closest to hour boundary)
 query = f"""
 WITH hourly_buckets AS (
 SELECT
 symbol,
 date_trunc('hour', timestamp) as hour,
 timestamp,
 mid_price,
 best_bid,
 best_ask,
 spread_pct,
 imbalance_10,
 imbalance_20,
 bid_depth_10,
 ask_depth_10,
 bids,
 asks,
 wall_detection,
 absorption_metrics,
 ROW_NUMBER OVER (
 PARTITION BY date_trunc('hour', timestamp)
 ORDER BY ABS(EXTRACT(MINUTE FROM timestamp) * 60 + EXTRACT(SECOND FROM timestamp))
 ) as rn
 FROM {self.schema}.orderbooks
 WHERE symbol = %s
 AND timestamp >= %s
 AND timestamp <= %s
 )
 SELECT
 symbol,
 hour,
 timestamp,
 mid_price,
 best_bid,
 best_ask,
 spread_pct,
 imbalance_10,
 imbalance_20,
 bid_depth_10,
 ask_depth_10,
 bids,
 asks,
 wall_detection,
 absorption_metrics
 FROM hourly_buckets
 WHERE rn = 1
 ORDER BY hour ASC;
 """

 cur.execute(query, (symbol, start_time, end_time))
 rows = cur.fetchall

 logger.info(f"Fetched {len(rows)} hourly snapshots for {symbol} (requested {hours} hours)")
 return [dict(row) for row in rows]

 def get_snapshot_count(self, symbol: Optional[str] = None) -> int:
 """
 Get total count of snapshots

 Args:
 symbol: Optional symbol filter

 Returns:
 Total count of snapshots
 """
 self.connect

 with self.conn.cursor as cur:
 if symbol:
 query = f"SELECT COUNT(*) FROM {self.schema}.orderbooks WHERE symbol = %s;"
 cur.execute(query, (symbol,))
 else:
 query = f"SELECT COUNT(*) FROM {self.schema}.orderbooks;"
 cur.execute(query)

 count = cur.fetchone[0]
 logger.info(f"Snapshot count{' for ' + symbol if symbol else ''}: {count}")
 return count

 def get_symbols(self) -> List[str]:
 """
 Get list of symbols with data

 Returns:
 List of unique symbols
 """
 self.connect

 with self.conn.cursor as cur:
 query = f"SELECT DISTINCT symbol FROM {self.schema}.orderbooks ORDER BY symbol;"
 cur.execute(query)
 symbols = [row[0] for row in cur.fetchall]

 logger.info(f"Found {len(symbols)} symbols: {symbols}")
 return symbols

 def get_time_range(self, symbol: Optional[str] = None) -> Dict[str, datetime]:
 """
 Get min/max timestamps

 Args:
 symbol: Optional symbol filter

 Returns:
 Dict with 'min_time' and 'max_time'
 """
 self.connect

 with self.conn.cursor as cur:
 if symbol:
 query = f"""
 SELECT MIN(timestamp), MAX(timestamp)
 FROM {self.schema}.orderbooks
 WHERE symbol = %s;
 """
 cur.execute(query, (symbol,))
 else:
 query = f"""
 SELECT MIN(timestamp), MAX(timestamp)
 FROM {self.schema}.orderbooks;
 """
 cur.execute(query)

 min_time, max_time = cur.fetchone

 logger.info(f"Time range{' for ' + symbol if symbol else ''}: {min_time} to {max_time}")
 return {
 'min_time': min_time,
 'max_time': max_time,
 }


# Example usage
if __name__ == '__main__':
 logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

 print("=" * 80)
 print("Orderbook Query Layer Test")
 print("=" * 80)

 try:
 with OrderbookQuery as query:
 # Get symbols
 print("\n1. Available Symbols:")
 symbols = query.get_symbols
 print(f" {symbols}")

 # Get count
 print("\n2. Total Snapshots:")
 count = query.get_snapshot_count
 print(f" {count} snapshots")

 # Get time range
 print("\n3. Time Range:")
 time_range = query.get_time_range
 print(f" From: {time_range['min_time']}")
 print(f" To: {time_range['max_time']}")

 # Get latest snapshots
 print("\n4. Latest Snapshots:")
 latest = query.get_latest_snapshots(limit=2)
 for snap in latest:
 print(f"\n {snap['symbol']} @ {snap['timestamp']}:")
 print(f" Spread: {snap['spread_pct']}%")
 print(f" Imbalance: {snap['imbalance_10']}")
 print(f" Bid Depth: {snap['bid_depth_10']}")
 print(f" Ask Depth: {snap['ask_depth_10']}")

 print("\n" + "=" * 80)
 print("✅ Query Layer Test Completed Successfully!")
 print("=" * 80)

 except Exception as e:
 print(f"\n❌ Test Failed: {e}")
 import traceback
 traceback.print_exc
