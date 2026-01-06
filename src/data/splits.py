"""Dataset splitting utilities."""

import random
from enum import Enum

import numpy as np
import ray.data as rd
from loguru import logger

from src.config.constants import DatasetSplit


class SplitStrategy(str, Enum):
    """Dataset split strategies."""

    RANDOM = "random"
    STRATIFIED = "stratified"
    TIME_BASED = "time_based"
    GROUP_BASED = "group_based"


class DatasetSplitter:
    """Dataset splitter with various strategies."""

    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        strategy: SplitStrategy = SplitStrategy.RANDOM,
        seed: int = 42,
        stratify_col: str | None = None,
        time_col: str | None = None,
        group_col: str | None = None,
    ):
        """
        Initialize dataset splitter.

        Args:
            train_size: Training set proportion.
            val_size: Validation set proportion.
            test_size: Test set proportion.
            strategy: Split strategy.
            seed: Random seed.
            stratify_col: Column for stratified splitting.
            time_col: Column for time-based splitting.
            group_col: Column for group-based splitting.

        Raises:
            ValueError: If proportions don't sum to 1.
        """
        total = train_size + val_size + test_size
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split proportions must sum to 1, got {total}")

        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.strategy = strategy
        self.seed = seed
        self.stratify_col = stratify_col
        self.time_col = time_col
        self.group_col = group_col

        # Set random seed
        random.seed(seed)
        np.random.seed(seed)

    def split(
        self, dataset: rd.Dataset, split_col: str = "split"
    ) -> dict[DatasetSplit, rd.Dataset]:
        """
        Split dataset into train/val/test.

        Args:
            dataset: Dataset to split.
            split_col: Column to store split labels.

        Returns:
            Dictionary of split datasets.

        Raises:
            ValueError: If strategy requirements not met.
        """
        logger.info(
            f"Splitting dataset with strategy: {self.strategy}, "
            f"split: {self.train_size}/{self.val_size}/{self.test_size}"
        )

        if self.strategy == SplitStrategy.RANDOM:
            return self._split_random(dataset, split_col)
        elif self.strategy == SplitStrategy.STRATIFIED:
            return self._split_stratified(dataset, split_col)
        elif self.strategy == SplitStrategy.TIME_BASED:
            return self._split_time_based(dataset, split_col)
        elif self.strategy == SplitStrategy.GROUP_BASED:
            return self._split_group_based(dataset, split_col)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _split_random(
        self, dataset: rd.Dataset, split_col: str
    ) -> dict[DatasetSplit, rd.Dataset]:
        """Split dataset randomly."""
        # Use Ray Data's built-in split
        train_test_split = self.train_size + self.val_size
        remaining_split = self.val_size / (self.val_size + self.test_size)

        # First split: train vs (val + test)
        train_ds, remaining_ds = dataset.train_test_split(
            test_size=1 - self.train_size, seed=self.seed
        )

        # Second split: val vs test
        val_ds, test_ds = remaining_ds.train_test_split(
            test_size=self.test_size / (self.val_size + self.test_size), seed=self.seed
        )

        # Add split labels
        train_ds = train_ds.map_batches(
            lambda df: self._add_split_label(df, DatasetSplit.TRAIN, split_col),
            batch_format="pandas",
        )
        val_ds = val_ds.map_batches(
            lambda df: self._add_split_label(df, DatasetSplit.VALIDATION, split_col),
            batch_format="pandas",
        )
        test_ds = test_ds.map_batches(
            lambda df: self._add_split_label(df, DatasetSplit.TEST, split_col),
            batch_format="pandas",
        )

        return {
            DatasetSplit.TRAIN: train_ds,
            DatasetSplit.VALIDATION: val_ds,
            DatasetSplit.TEST: test_ds,
        }

    def _split_stratified(
        self, dataset: rd.Dataset, split_col: str
    ) -> dict[DatasetSplit, rd.Dataset]:
        """Split dataset with stratification."""
        if self.stratify_col is None:
            raise ValueError("stratify_col must be specified for stratified split")

        # For simplicity, we'll use random split for now
        # In production, you might want to implement proper stratification
        logger.warning("Stratified split not fully implemented, using random split")
        return self._split_random(dataset, split_col)

    def _split_time_based(
        self, dataset: rd.Dataset, split_col: str
    ) -> dict[DatasetSplit, rd.Dataset]:
        """Split dataset based on time."""
        if self.time_col is None:
            raise ValueError("time_col must be specified for time-based split")

        # Sort by time column
        sorted_ds = dataset.sort(self.time_col)

        # Calculate split indices
        total = sorted_ds.count()
        train_end = int(total * self.train_size)
        val_end = train_end + int(total * self.val_size)

        # Split using indices (this is simplified)
        # In production, use proper window-based splitting
        splits = {}
        current_start = 0

        for split_name, split_size in [
            (DatasetSplit.TRAIN, train_end),
            (DatasetSplit.VALIDATION, val_end - train_end),
            (DatasetSplit.TEST, total - val_end),
        ]:
            # This is a simplified approach
            # In practice, you'd want to use window-based sampling
            split_ds = sorted_ds.limit(split_size, offset=current_start)
            split_ds = split_ds.map_batches(
                lambda df: self._add_split_label(df, split_name, split_col),
                batch_format="pandas",
            )
            splits[split_name] = split_ds
            current_start += split_size

        return splits

    def _split_group_based(
        self, dataset: rd.Dataset, split_col: str
    ) -> dict[DatasetSplit, rd.Dataset]:
        """Split dataset by groups."""
        if self.group_col is None:
            raise ValueError("group_col must be specified for group-based split")

        # Get unique groups
        groups = dataset.unique(self.group_col)
        unique_groups = [row[self.group_col] for row in groups.take(groups.count())]

        # Shuffle groups
        random.shuffle(unique_groups)

        # Split groups
        num_groups = len(unique_groups)
        train_groups_end = int(num_groups * self.train_size)
        val_groups_end = train_groups_end + int(num_groups * self.val_size)

        train_groups = set(unique_groups[:train_groups_end])
        val_groups = set(unique_groups[train_groups_end:val_groups_end])
        test_groups = set(unique_groups[val_groups_end:])

        # Create splits based on groups
        splits = {}
        for split_name, group_set in [
            (DatasetSplit.TRAIN, train_groups),
            (DatasetSplit.VALIDATION, val_groups),
            (DatasetSplit.TEST, test_groups),
        ]:
            split_ds = dataset.filter(lambda row: row[self.group_col] in group_set)
            split_ds = split_ds.map_batches(
                lambda df: self._add_split_label(df, split_name, split_col),
                batch_format="pandas",
            )
            splits[split_name] = split_ds

        return splits

    @staticmethod
    def _add_split_label(df, split_name: DatasetSplit, split_col: str):
        """Add split label to DataFrame."""
        df = df.copy()
        df[split_col] = split_name.value
        return df


def create_default_splitter() -> DatasetSplitter:
    """
    Create default dataset splitter.

    Returns:
        DatasetSplitter instance.
    """
    from src.config import settings

    return DatasetSplitter(
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        strategy=SplitStrategy.RANDOM,
        seed=settings.seed if hasattr(settings, 'seed') else 42,
    )
