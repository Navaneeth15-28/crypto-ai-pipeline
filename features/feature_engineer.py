from collections import deque

import numpy as np


class StreamingFeatureEngine:

    def __init__(self, window_size_seconds=60):

        self.window_size = window_size_seconds

        # Each trade:
        # timestamp, price, volume, is_buyer_maker
        self.history = deque()

    # ========================================================
    # UPDATE STREAM
    # ========================================================

    def update_and_get_features(
        self,
        current_price,
        current_volume,
        current_ts,
        is_buyer_maker=None
    ):

        self.history.append(
            (
                current_ts,
                current_price,
                current_volume,
                is_buyer_maker
            )
        )

        cutoff_time = (
            current_ts
            - self.window_size
        )

        while (
            self.history
            and self.history[0][0] < cutoff_time
        ):

            self.history.popleft()

        if len(self.history) < 2:

            return None

        # ====================================================
        # ARRAYS
        # ====================================================

        timestamps = np.array(
            [x[0] for x in self.history],
            dtype=float
        )

        prices = np.array(
            [x[1] for x in self.history],
            dtype=float
        )

        volumes = np.array(
            [x[2] for x in self.history],
            dtype=float
        )

        maker_flags = [
            x[3]
            for x in self.history
        ]

        # ====================================================
        # TIME WINDOWS
        # ====================================================

        prices_5 = prices[
            timestamps >= current_ts - 5
        ]

        volumes_5 = volumes[
            timestamps >= current_ts - 5
        ]

        prices_10 = prices[
            timestamps >= current_ts - 10
        ]

        volumes_10 = volumes[
            timestamps >= current_ts - 10
        ]

        prices_30 = prices[
            timestamps >= current_ts - 30
        ]

        volumes_30 = volumes[
            timestamps >= current_ts - 30
        ]

        prices_60 = prices
        volumes_60 = volumes

        # ====================================================
        # BASIC PRICE/VOLUME FEATURES
        # ====================================================

        features = {}

        # ----------------------------------------------------
        # 5 seconds
        # ----------------------------------------------------

        features[
            "rolling_mean_price_5s"
        ] = float(
            np.mean(prices_5)
        )

        features[
            "rolling_std_price_5s"
        ] = float(
            np.std(prices_5)
        )

        features[
            "rolling_volume_5s"
        ] = float(
            np.sum(volumes_5)
        )

        features[
            "trade_frequency_5s"
        ] = int(
            len(prices_5)
        )

        if len(prices_5) >= 2:

            features[
                "price_return_5s"
            ] = float(
                (
                    (
                        current_price
                        - prices_5[0]
                    )
                    / prices_5[0]
                )
                * 100
            )

        else:

            features[
                "price_return_5s"
            ] = 0.0

        # ----------------------------------------------------
        # 10 seconds
        # ----------------------------------------------------

        features[
            "rolling_mean_price_10s"
        ] = float(
            np.mean(prices_10)
        )

        features[
            "rolling_std_price_10s"
        ] = float(
            np.std(prices_10)
        )

        features[
            "rolling_volume_10s"
        ] = float(
            np.sum(volumes_10)
        )

        features[
            "trade_frequency_10s"
        ] = int(
            len(prices_10)
        )

        if len(prices_10) >= 2:

            features[
                "price_return_10s"
            ] = float(
                (
                    (
                        current_price
                        - prices_10[0]
                    )
                    / prices_10[0]
                )
                * 100
            )

        else:

            features[
                "price_return_10s"
            ] = 0.0

        features[
            "price_range_10s"
        ] = float(
            np.max(prices_10)
            - np.min(prices_10)
        )

        if len(prices_10) >= 2:

            features[
                "volatility_10s"
            ] = float(
                np.std(
                    np.diff(prices_10)
                    / prices_10[:-1]
                )
            )

        else:

            features[
                "volatility_10s"
            ] = 0.0

        # ----------------------------------------------------
        # 30 seconds
        # ----------------------------------------------------

        features[
            "rolling_mean_price_30s"
        ] = float(
            np.mean(prices_30)
        )

        features[
            "rolling_std_price_30s"
        ] = float(
            np.std(prices_30)
        )

        features[
            "rolling_volume_30s"
        ] = float(
            np.sum(volumes_30)
        )

        features[
            "trade_frequency_30s"
        ] = int(
            len(prices_30)
        )

        if len(prices_30) >= 2:

            return_30 = (
                (
                    current_price
                    - prices_30[0]
                )
                / prices_30[0]
            ) * 100

            features[
                "price_return_30s"
            ] = float(
                return_30
            )

            features[
                "price_momentum_30s"
            ] = float(
                return_30
            )

        else:

            features[
                "price_return_30s"
            ] = 0.0

            features[
                "price_momentum_30s"
            ] = 0.0

        features[
            "price_range_30s"
        ] = float(
            np.max(prices_30)
            - np.min(prices_30)
        )

        if len(prices_30) >= 2:

            features[
                "volatility_30s"
            ] = float(
                np.std(
                    np.diff(prices_30)
                    / prices_30[:-1]
                )
            )

        else:

            features[
                "volatility_30s"
            ] = 0.0

        # ----------------------------------------------------
        # 60 seconds
        # ----------------------------------------------------

        features[
            "rolling_mean_price_60s"
        ] = float(
            np.mean(prices_60)
        )

        features[
            "rolling_std_price_60s"
        ] = float(
            np.std(prices_60)
        )

        features[
            "rolling_volume_60s"
        ] = float(
            np.sum(volumes_60)
        )

        features[
            "trade_frequency_60s"
        ] = int(
            len(prices_60)
        )

        if len(prices_60) >= 2:

            features[
                "price_momentum_60s"
            ] = float(
                (
                    (
                        current_price
                        - prices_60[0]
                    )
                    / prices_60[0]
                )
                * 100
            )

        else:

            features[
                "price_momentum_60s"
            ] = 0.0

        features[
            "price_range_60s"
        ] = float(
            np.max(prices_60)
            - np.min(prices_60)
        )

        # ====================================================
        # MOMENTUM FEATURES
        # ====================================================

        features[
            "momentum_acceleration"
        ] = float(
            features["price_return_10s"]
            - features["price_return_30s"]
        )

        # ====================================================
        # VOLUME ACCELERATION
        # ====================================================

        volume_5 = np.sum(
            volumes_5
        )

        volume_30 = np.sum(
            volumes_30
        )

        features[
            "volume_acceleration"
        ] = float(
            volume_5
            - (volume_30 / 6.0)
        )

        # ====================================================
        # TRADE FREQUENCY CHANGE
        # ====================================================

        expected_frequency = (
            len(prices_30)
            / 6.0
        )

        features[
            "trade_frequency_change"
        ] = float(
            len(prices_5)
            - expected_frequency
        )

        # ====================================================
        # TRADE-SIDE MICROSTRUCTURE
        #
        # Binance:
        # is_buyer_maker = True
        #     -> buyer was maker
        #
        # is_buyer_maker = False
        #     -> buyer was taker
        #
        # For directional pressure we treat:
        #
        # False -> aggressive BUY volume
        # True  -> aggressive SELL volume
        # ====================================================

        def calculate_trade_side(
            window_start
        ):

            buy_volume = 0.0
            sell_volume = 0.0

            buy_trades = 0
            sell_trades = 0

            for (
                ts,
                price,
                volume,
                maker
            ) in self.history:

                if ts < window_start:
                    continue

                if maker is None:
                    continue

                if maker:

                    sell_volume += volume
                    sell_trades += 1

                else:

                    buy_volume += volume
                    buy_trades += 1

            total_volume = (
                buy_volume
                + sell_volume
            )

            total_trades = (
                buy_trades
                + sell_trades
            )

            if total_volume > 0:

                volume_imbalance = (
                    buy_volume
                    - sell_volume
                ) / total_volume

                buy_sell_ratio = (
                    buy_volume
                    / sell_volume
                    if sell_volume > 0
                    else buy_volume
                )

            else:

                volume_imbalance = 0.0
                buy_sell_ratio = 0.0

            if total_trades > 0:

                trade_imbalance = (
                    buy_trades
                    - sell_trades
                ) / total_trades

            else:

                trade_imbalance = 0.0

            return {
                "buy_volume": buy_volume,
                "sell_volume": sell_volume,
                "buy_trades": buy_trades,
                "sell_trades": sell_trades,
                "volume_imbalance": volume_imbalance,
                "trade_imbalance": trade_imbalance,
                "buy_sell_ratio": buy_sell_ratio
            }

        # ====================================================
        # 5 SECOND TRADE-SIDE FEATURES
        # ====================================================

        side_5 = calculate_trade_side(
            current_ts - 5
        )

        features[
            "buy_volume_5s"
        ] = float(
            side_5["buy_volume"]
        )

        features[
            "sell_volume_5s"
        ] = float(
            side_5["sell_volume"]
        )

        features[
            "buy_trade_count_5s"
        ] = int(
            side_5["buy_trades"]
        )

        features[
            "sell_trade_count_5s"
        ] = int(
            side_5["sell_trades"]
        )

        features[
            "volume_imbalance_5s"
        ] = float(
            side_5["volume_imbalance"]
        )

        features[
            "trade_imbalance_5s"
        ] = float(
            side_5["trade_imbalance"]
        )

        features[
            "buy_sell_ratio_5s"
        ] = float(
            side_5["buy_sell_ratio"]
        )

        # ====================================================
        # 10 SECOND TRADE-SIDE FEATURES
        # ====================================================

        side_10 = calculate_trade_side(
            current_ts - 10
        )

        features[
            "buy_volume_10s"
        ] = float(
            side_10["buy_volume"]
        )

        features[
            "sell_volume_10s"
        ] = float(
            side_10["sell_volume"]
        )

        features[
            "buy_trade_count_10s"
        ] = int(
            side_10["buy_trades"]
        )

        features[
            "sell_trade_count_10s"
        ] = int(
            side_10["sell_trades"]
        )

        features[
            "volume_imbalance_10s"
        ] = float(
            side_10["volume_imbalance"]
        )

        features[
            "trade_imbalance_10s"
        ] = float(
            side_10["trade_imbalance"]
        )

        features[
            "buy_sell_ratio_10s"
        ] = float(
            side_10["buy_sell_ratio"]
        )

        # ====================================================
        # 30 SECOND TRADE-SIDE FEATURES
        # ====================================================

        side_30 = calculate_trade_side(
            current_ts - 30
        )

        features[
            "buy_volume_30s"
        ] = float(
            side_30["buy_volume"]
        )

        features[
            "sell_volume_30s"
        ] = float(
            side_30["sell_volume"]
        )

        features[
            "buy_trade_count_30s"
        ] = int(
            side_30["buy_trades"]
        )

        features[
            "sell_trade_count_30s"
        ] = int(
            side_30["sell_trades"]
        )

        features[
            "volume_imbalance_30s"
        ] = float(
            side_30["volume_imbalance"]
        )

        features[
            "trade_imbalance_30s"
        ] = float(
            side_30["trade_imbalance"]
        )

        features[
            "buy_sell_ratio_30s"
        ] = float(
            side_30["buy_sell_ratio"]
        )

        # ====================================================
        # 60 SECOND TRADE-SIDE FEATURES
        # ====================================================

        side_60 = calculate_trade_side(
            current_ts - 60
        )

        features[
            "buy_volume_60s"
        ] = float(
            side_60["buy_volume"]
        )

        features[
            "sell_volume_60s"
        ] = float(
            side_60["sell_volume"]
        )

        features[
            "buy_trade_count_60s"
        ] = int(
            side_60["buy_trades"]
        )

        features[
            "sell_trade_count_60s"
        ] = int(
            side_60["sell_trades"]
        )

        features[
            "volume_imbalance_60s"
        ] = float(
            side_60["volume_imbalance"]
        )

        features[
            "trade_imbalance_60s"
        ] = float(
            side_60["trade_imbalance"]
        )

        features[
            "buy_sell_ratio_60s"
        ] = float(
            side_60["buy_sell_ratio"]
        )

        return features