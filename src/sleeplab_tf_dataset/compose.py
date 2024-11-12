"""Tools for splitting and combining multiple SLF datasets."""
import logging
import numpy as np
import sleeplab_format as slf
import sleeplab_tf_dataset as sds
import tensorflow as tf

from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def parse_dataset_configs(cfg: dict[str, Any]) -> dict[str, sds.config.DatasetConfig]:
    """Parse configs for datasets and splits from dicts to DatasetConfigs."""
    cfg = cfg.copy()  # Copy so that the original config does not get modified
    splits = cfg.pop('splits')
    res = {}

    for k, split in splits.items():
        _cfg_dict = cfg.copy()
        _cfg_dict.update(split['config'])
        _cfg = sds.config.DatasetConfig.model_validate(_cfg_dict)
        res[k] = _cfg

    return res


def load_split(cfg: dict[str, Any], seed: int = 42) -> dict[str, tf.data.Dataset]:
    """Load and split a single series."""
    logger.info(f'Reading the series {cfg["series_name"]}...')
    slf_ds = slf.reader.read_dataset(
        ds_dir = Path(cfg['ds_dir']),
        series_names=[cfg['series_name']]
    )

    logger.info('Creating the splits...')
    # Read the subject IDs
    subj_ids = list(slf_ds.series[cfg['series_name']].subjects.keys())

    # Assert that the sum of split sizes matches the total number of subjects
    sum_split_sizes = sum([split['n'] for split in cfg['splits'].values()])
    _msg = f'Sum of split sizes does not match dataset size ({sum_split_sizes} != {len(subj_ids)})'
    assert sum_split_sizes == len(subj_ids), _msg

    # Randomly shuffle before splitting
    rng = np.random.default_rng(seed)
    permuted_subj_ids = rng.permutation(subj_ids)

    # Split the subject IDs
    split_subj_ids = {}
    curr_split_start = 0
    for k, split in cfg['splits'].items():
        curr_split_end = curr_split_start + split['n']
        split_subj_ids[k] = permuted_subj_ids[curr_split_start:curr_split_end]
        curr_split_start = curr_split_end

    # Parse config for each split
    split_cfgs = parse_dataset_configs(cfg)
    
    # Create the datasets for each split
    res = {}
    for k, cfg in split_cfgs.items():
        res[k] = sds.dataset.from_slf_dataset(
            slf_ds, cfg, subject_ids=split_subj_ids[k])
    
    return res


def load_split_concat(cfgs: dict[str, Any], seed: int = 42) -> dict[str, tf.data.Dataset]:
    """Load, split, and concatenate multiple SLF datasets.
    
    Args:
        cfgs: A dict with entries for each dataset.
        seed: The seed for random splitting.
    
    Returns:
        A dict of tf Datasets with entries for each split.
    """
    def _concat_ds_list(_ds_list):
        # If there's only one dataset, return it
        if len(_ds_list) == 1:
            return _ds_list[0]
        
        ds_concat = _ds_list[0]
        for ds in _ds_list[1:]:
            ds_concat = ds_concat.concatenate(ds)

        return ds_concat
    
    ds_list = [load_split(cfgs[k], seed=seed) for k in cfgs.keys()]
    split_keys = set([k for ds in ds_list for k in ds.keys()])
    datasets = {k: _concat_ds_list([ds[k] for ds in ds_list if k in ds.keys()]) for k in split_keys}

    return datasets
