from src.data_loader.dataloader import StockDataset
from src.data_loader.utils import *


if __name__ == "__main__":

    cache_path = r"../../data/torch_datasets/window.pt"
    dataset_window = StockDataset(df, mode="window", sequence_length=64, cache_path=cache_path)

    cache_path = r"../../data/torch_datasets/autoreg.pt"
    dataset_autoreg = StockDataset(df, mode="autoregressive", sequence_length=64, cache_path=cache_path)
