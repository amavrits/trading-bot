import pandas as pd
from typing import Optional, Tuple, Union, List, Literal
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class StockDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        target_features: Optional[List[str]] = ["Close"],
        sequence_length: Optional[int] = 64,
        mode: Literal["window", "autoregressive"] = "window",
        binary_target: bool = False,
        cache_path: Optional[str | Path] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            df: DataFrame from load_multi_stock_data
            features: Input features used in sequences
            target_features: Columns to use as y (defaults to features)
            sequence_length: Length of input sequences (X)
            mode:  Supports two modes:
                - "window": fixed-length sliding windows (X → y)
                - "autoregressive": full sequence X → shifted target y
            binary_target: Returns a +1 when the price rises from the last timestep and 0 when it drops.
            device: Optional torch device
        """
        self.mode = mode
        self.device = device
        self.sequence_length = sequence_length
        self.binary_target = binary_target
        self.X, self.y = [], []

        # Auto-infer feature columns if not specified
        exclude = {"Date", "Ticker"}
        if features is None:
            features = [col for col in df.columns if col not in exclude and pd.api.types.is_numeric_dtype(df[col])]
        if target_features is None:
            target_features = features

        self.features = features
        self.target_features = target_features

        # ✅ Load from cache if available
        if not isinstance(cache_path, Path): cache_path = Path(cache_path)
        if cache_path and cache_path.is_file():
            print(f"✅ Loading cached dataset from {cache_path}")
            data = torch.load(cache_path)
            self.X, self.y = data["X"], data["y"]
            if device:
                self.X = [x.to(device) for x in self.X]
                self.y = [y.to(device) for y in self.y]
            return  # done

        # Group by ticker
        for _, df_ticker in df.groupby("Ticker"):

            df_ticker = df_ticker.sort_values("Date")
            data = df_ticker[features].values

            if self.binary_target:
                # Generate binary targets: 1 if Close > Open, else 0
                y_binary = (df_ticker["Close"].values > df_ticker["Open"].values).astype(float).reshape(-1, 1)
                targets = y_binary
            else:
                targets = df_ticker[self.target_features].values

            if len(data) <= sequence_length:
                continue

            if self.mode == "window":
                for i in range(len(data) - sequence_length):
                    x_window = data[i:i + sequence_length]
                    y_next = targets[i + sequence_length]
                    self.X.append(torch.tensor(x_window, dtype=torch.float32))
                    self.y.append(torch.tensor(y_next, dtype=torch.float32))

            elif self.mode == "autoregressive":
                for i in range(len(data) - sequence_length):
                    x_seq = data[i: i + sequence_length]
                    y_seq = targets[i + 1: i + sequence_length + 1]
                    self.X.append(torch.tensor(x_seq, dtype=torch.float32))
                    self.y.append(torch.tensor(y_seq, dtype=torch.float32))

        if device:
            self.X = [x.to(device) for x in self.X]
            self.y = [y.to(device) for y in self.y]

        if cache_path:
            print(f"💾 Saving dataset cache to {cache_path}")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"X": self.X, "y": self.y}, cache_path)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_stock_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":

    pass

