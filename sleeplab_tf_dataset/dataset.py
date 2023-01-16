import json  # TODO: faster alternatives? e.g. ultrajson
import numpy as np
import pandas as pd
import sleeplab_format as slf
import tensorflow as tf

from functools import partial
from pathlib import Path
from sleeplab_tf_dataset.config import ComponentConfig, DatasetConfig
from typing import Any, Callable


def load_sample_array(
        subject_dir: tf.Tensor,
        start_sec: tf.Tensor,
        duration: tf.Tensor,
        cfg: dict[str, Any],
        dtype: tf.DType = tf.float32) -> tf.Tensor:
    """Load sample array to tf.Tensor.
    
    Args:
        subject_dir: The subject directory path as a string Tensor
        start_sec: The first timepoint to load as seconds from start of recording
        duration: The duration to load. If duration == -1, load until the end
        cfg: The ComponentConfig to load
        dtype: Tensorflow DType for the returned signal

    Returns:
        The signal as a tf.Tensor
    """
    _subj_dir = Path(subject_dir.numpy().decode())
    fs = cfg['fs']
    start_idx = start_sec * fs
    end_idx = start_idx + duration * fs

    start_idx = tf.cast(start_idx, tf.int32)
    end_idx = tf.cast(end_idx, tf.int32)

    arr_fpath = _subj_dir / cfg['src_name'] / 'data.npy'
    s = np.load(arr_fpath, mmap_mode='r')
    s = s[start_idx:end_idx]

    return tf.convert_to_tensor(s[..., np.newaxis], dtype=dtype)


def load_annotations(
        subject_dir: tf.Tensor,
        start_sec: tf.Tensor,
        duration: tf.Tensor,
        cfg: dict[str, Any]) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor]:
    """Load sleeplab-format Annotations as tf.Tensor data.

    Args:
        cfg.return_type: 'segmentation_combined', 'segmentation_separate', or 'bbox'.
            segmentation_combined returns a 1-d array where the values looked up
                from cfg.value_map. Any overlap in annotations leads to inconsistencies.
            segmentation_separate returns a Tensor containing separate arrays of 0/1
                for each entry in cfg.value_map, as well as the corresponding labels for each row
            bbox returns a tuple of two tensors; first contains the bbox coordinates and
                the second contains the corresponding labels from cfg.value_map
    """
    _subj_dir = Path(subject_dir.numpy().decode())
    fpath = _subj_dir / cfg['src_name']
    df = pd.read_parquet(fpath)

    frame_start_sec = start_sec
    frame_duration = duration

    if cfg['fs'] is not None:
        fs = cfg['fs']
    else:
        fs = 1 / cfg['sampling_interval']

    # Calculate end times
    df['end_sec'] = df['start_sec'] + df['duration']

    # Filter out events outside the desired timeframe
    frame_end_sec = frame_start_sec + frame_duration
    df = df[df['end_sec'] > frame_start_sec]
    df = df[df['start_sec'] < frame_end_sec]

    # Filter the desired event types
    df = df[df['name'].isin(cfg['value_map'].keys())]

    # Transform start and end times from seconds to indices wrt the timeframe
    df['start_idx'] = ((df['start_sec'] - frame_start_sec) * fs).astype(int)
    df['start_idx'] = df['start_idx'].clip(lower=0)

    df['end_idx'] = ((df['end_sec'] - frame_start_sec) * fs).astype(int)
    df['end_idx'] = df['end_idx'].clip(upper=int(fs*duration))

    # Transform the dataframe to the desired output format
    if cfg['return_type'] == 'segmentation_combined':
        res = np.full(int(frame_duration * fs), cfg['value_map']['_default'])
        for _, row in df.iterrows():
            res[row.start_idx:row.end_idx] = cfg['value_map'][row['name']]

        return tf.convert_to_tensor(res, dtype=tf.int32)
    
    elif cfg['return_type'] == 'segmentation_separate':
        res = {v: np.full(int(frame_duration * fs), cfg['value_map']['_default']) for v in set(cfg['value_map'].values())}
        for _, row in df.iterrows():
            res[cfg['value_map'][row['name']]][row.start_idx:row.end_idx] = 1
    
        for k, v in res.items():
            res[k] = tf.convert_to_tensor(v, dtype=tf.int32)

        return tf.stack(list(res.values())), tf.convert_to_tensor(list(res.keys()))

    elif cfg['return_type'] == 'bbox':
        bboxes = []
        labels = []
        for _, row in df.iterrows():
            bboxes.append([row.start_idx, row.end_idx])
            labels.append(cfg['value_map'][row['name']])

        bboxes = tf.convert_to_tensor(bboxes, dtype=tf.int32)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)

        return bboxes, labels        

    else:
        raise AttributeError(f'Unsupported return_type {cfg["return_type"]}')


@tf.function
def load_component(
        subject_dir: tf.Tensor,
        start_sec: tf.Tensor,
        duration: tf.Tensor,
        orig_duration: float,
        cfg: dict[str, Any]) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor]:
    """Load a single component of an element.

    Args:
        orig_duration: The original duration from the configuration.
            This is needed to set the shape of the loaded component.
    """
    # Compute the shape for first dimension of the output (excluding batch dim)."""
    if cfg['fs'] is not None:
        fs = cfg['fs']
    else:
        fs = 1 / cfg['sampling_interval']

    shape_0 = int(orig_duration * fs)

    if cfg['ctype'] == 'sample_array':
        res = tf.py_function(
            partial(load_sample_array, cfg=cfg),
            [subject_dir, start_sec, duration],
            tf.float32
        )
        if orig_duration == -1.0:
            res.set_shape([None, 1])
        else:
            res.set_shape([shape_0, 1])
        return res

    elif cfg['ctype'] == 'annotation':

        if cfg['return_type'] == 'segmentation_combined':
            pyfunc_return_type = tf.int32
            res = tf.py_function(
                partial(load_annotations, cfg=cfg),
                [subject_dir, start_sec, duration],
                pyfunc_return_type
            )
            if orig_duration == -1.0:
                res.set_shape((None,))
            else:
                res.set_shape((shape_0,))

            return res

        elif cfg['return_type'] == 'segmentation_separate':
            pyfunc_return_type = [tf.int32, tf.int32]
            nclasses = len(set(cfg['value_map'].values()))

            res = tf.py_function(
                partial(load_annotations, cfg=cfg),
                [subject_dir, start_sec, duration],
                pyfunc_return_type
            )
            if orig_duration == -1.0:
                res[0].set_shape([None, nclasses])
            else:
                res[0].set_shape([shape_0, nclasses])
            res[1].set_shape((nclasses,))

            return res

        else:
            # bboxes
            pyfunc_return_type = [tf.int32, tf.int32]

            res = tf.py_function(
                partial(load_annotations, cfg=cfg),
                [subject_dir, start_sec, duration],
                pyfunc_return_type
            )
            res[0].set_shape([None, 2])
            res[1].set_shape((None,))

            return res
    
    else:
        raise AttributeError(f'Unknown component type {cfg["ctype"]}')


@tf.function
def load_element(
        subject_dir: tf.Tensor,
        roi_start_sec: tf.Tensor,
        roi_end_sec: tf.Tensor,
        start_sec: float,
        duration: float,
        component_cfgs: dict[str, dict[str, Any]],
        start_sec_sampling_interval: tf.Tensor = tf.constant(30.0)) -> dict[str, tf.Tensor]:
    """Load a single element (subject) according to component configs.
    
    Args:
        roi_start_sec: The start time of the region of interest to be used
            in seconds from recording_start
        roi_end_sec: The end time of the region of interest to be used
            in seconds from recording_start
        start_sec_sampling_interval: The interval for start_sec sampling in seconds.
            Defaults to 30, meaning that the start_sec will be sampled from 30-s epochs.
    """
    res = {}

    start_sec = tf.cast(start_sec, tf.float32)
    # Randomly sample the start time of the frame if start_sec is not specified
    if start_sec == -1.0:
        #assert duration > tf.constant(0.0), 'Need to define start_sec or duration'
        start_interval = tf.random.uniform(shape=[],
            minval=tf.cast(roi_start_sec, tf.int32),
            maxval=tf.cast((roi_end_sec - duration) / start_sec_sampling_interval, tf.int32) + 1,
            dtype=tf.int32)
        
        start_sec = tf.cast(start_interval, tf.float32) * start_sec_sampling_interval

    # Load until the end if duration is not specified
    if duration == -1.0:
        tf_duration = tf.cast(roi_end_sec - start_sec, tf.float32)
    else:
        tf_duration = tf.cast(duration, tf.float32)

    for k, cfg in component_cfgs.items():
        component = load_component(subject_dir, start_sec, tf_duration, duration, cfg)
        
        if cfg['ctype'] == 'annotation' and cfg['return_type'] != 'segmentation_combined':
            component = {'values': component[0], 'labels': component[1]}
        
        res[k] = component

    return res


def roi_start_end_sec(
        subjects: dict[str, slf.models.Subject],
        src_type: str, src_name: str) -> tuple[float, float]:
    """Resolve start and end times for the region of interest.
    
    Args:
        series: The slf.models.Series
        src_type: The type of sleeplab-format entry from where the start and end are
            parsed. 'annotation' or 'sample_array'
        src_name: The name of the sleeplab-format entry. For example,
            'hypnogram' or 'C4'
    
    Returns:
        A tuple of lists with start and end times for each subject in seconds.
    """
    def roi_start_end_from_annotation(subject, src_name):
        ann_list = subject.annotations[src_name].annotations
        start = ann_list[0].start_sec
        end = ann_list[-1].start_sec + ann_list[-1].duration
        return start, end

    def roi_start_end_from_sample_array(subject, src_name):
        sarr = subject.sample_arrays[src_name]
        fs = sarr.attributes.sampling_rate
        slen = sarr.values_func().shape[0]
        return 0.0, float(slen / fs)

    assert src_type in ('analysis_start_end', 'annotation', 'sample_array')
    start_sec_list = []
    end_sec_list = []
    
    for _, subj in subjects.items():
        if src_type == 'annotation':
            start, end = roi_start_end_from_annotation(subj, src_name)
        elif src_type == 'sample_array':
            start, end = roi_start_end_from_sample_array(subj, src_name)
        elif src_type == 'analysis_start_end':
            # TODO This is not yet tested
            start = subj.attributes.analysis_start
            end = subj.attributes.analysis_end
        
        start_sec_list.append(start)
        end_sec_list.append(end)

    return start_sec_list, end_sec_list


def from_slf_dataset(
        slf_ds: slf.models.Dataset,
        cfg: DatasetConfig,
        subject_ids: list[str] | None = None) -> tf.data.Dataset:
    def component_config_to_dict(cfg):
        # Tensorflow does not understand custom objects so cast it to dict
        return {k: v.dict() for k, v in cfg.items()}

    # Resolve subject_dirs as string tensors
    series_dir = cfg.ds_dir / cfg.series_name
    subject_dirs = []
    series = slf_ds.series[cfg.series_name]
    subjects = series.subjects

    if subject_ids is not None:
        subjects = {k: v for k, v in subjects.items() if k in subject_ids}

    for subj_id in subjects.keys():
        subject_dir = series_dir / subj_id
        subj_dir_tensor = tf.convert_to_tensor(str(subject_dir))
        subject_dirs.append(subj_dir_tensor)

    # Resolve ROI start and end times
    roi_starts, roi_ends = roi_start_end_sec(subjects, cfg.roi_src_type, cfg.roi_src_name)

    # Create a dict with subject metadata and ROI information
    slf_ds_dict = {
        'subject_dir': subject_dirs,
        'roi_start_sec': roi_starts,
        'roi_end_sec': roi_ends
    }

    # Create a tf Dataset from the dict
    tf_ds = tf.data.Dataset.from_tensor_slices(slf_ds_dict)

    if cfg.start_sec == -1.0:
        assert cfg.duration > 0, 'Need to define start_sec or duration'

    # Create a partial func for mapping
    _map_func = partial(load_element,
        start_sec=cfg.start_sec,
        duration=cfg.duration,
        component_cfgs=component_config_to_dict(cfg.components))
    
    # Map the dataset
    tf_ds = tf_ds.map(lambda e: _map_func(e['subject_dir'], e['roi_start_sec'], e['roi_end_sec']))
    
    return tf_ds
