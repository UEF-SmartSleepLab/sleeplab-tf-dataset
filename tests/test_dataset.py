import numpy as np
import pytest
import sleeplab_format as slf
import tensorflow as tf

from functools import partial
from sleeplab_tf_dataset import config, dataset


def test_load_array_full(ds_dir, example_config_path):
    cfg = config.parse_config(example_config_path)
    component_cfg = cfg.components['c1']
    subject_dir = ds_dir / 'series1' / '10001'
    subject_dir = tf.convert_to_tensor(str(subject_dir))
    start_sec = tf.convert_to_tensor(0.0)
    duration = tf.convert_to_tensor(60.0)
    s = dataset.load_sample_array(subject_dir, start_sec, duration, component_cfg.dict())
    
    assert tuple(s.shape) == (duration*32, 1)
    assert s.dtype == tf.float32

def test_load_array_partial(ds_dir, example_config_path):
    cfg = config.parse_config(example_config_path)
    component_cfg = cfg.components['c1']
    subject_dir = ds_dir / 'series1' / '10001'
    subject_dir = tf.convert_to_tensor(str(subject_dir))
    start_sec = tf.convert_to_tensor(10.0)
    duration = tf.convert_to_tensor(25.0)
    s = dataset.load_sample_array(subject_dir, start_sec, duration, component_cfg.dict())
    
    assert tuple(s.shape) == (duration*32, 1)
    assert s.dtype == tf.float32


def test_load_hypnogram_full(ds_dir, example_config_path):
    cfg = config.parse_config(example_config_path)
    component_cfg = cfg.components['hypnogram']
    subject_dir = ds_dir / 'series1' / '10001'
    subject_dir = tf.convert_to_tensor(str(subject_dir))
    start_sec = tf.convert_to_tensor(0.0)
    duration = tf.convert_to_tensor(60.0)

    # Test segmentation_combined
    component_cfg.return_type = 'segmentation_combined'
    ann = dataset.load_annotations(subject_dir, start_sec, duration,
        component_cfg.dict())
    
    assert tuple(ann.shape) == (duration*(1/component_cfg.sampling_interval),)
    assert ann.numpy().tolist() == [1, 0]

    # Test segmentation_separate
    component_cfg.return_type = 'segmentation_separate'
    ann, labels = dataset.load_annotations(subject_dir, start_sec, duration,
        component_cfg.dict())
    assert set(labels.numpy().tolist()) == set([0, 1, 2, 3, 4])
    assert ann[0].numpy().tolist() == [0, 1]
    assert ann[1].numpy().tolist() == [1, 0]


def test_load_hypnogram_partial(ds_dir, example_config_path):
    cfg = config.parse_config(example_config_path)
    component_cfg = cfg.components['hypnogram']
    subject_dir = ds_dir / 'series1' / '10001'
    subject_dir = tf.convert_to_tensor(str(subject_dir))
    start_sec = tf.convert_to_tensor(30.0)
    duration = tf.convert_to_tensor(30.0)

    # Test segmentation_combined
    component_cfg.return_type = 'segmentation_combined'
    ann = dataset.load_annotations(subject_dir, start_sec, duration,
        component_cfg.dict())
    
    assert tuple(ann.shape) == (duration*(1/component_cfg.sampling_interval),)
    assert ann.numpy().tolist() == [0]


def test_load_events_full(ds_dir, example_config_path):
    cfg = config.parse_config(example_config_path)
    component_cfg = cfg.components['events']
    subject_dir = ds_dir / 'series1' / '10001'
    subject_dir = tf.convert_to_tensor(str(subject_dir))
    start_sec = tf.convert_to_tensor(0.0)
    duration = tf.convert_to_tensor(60.0)

    # Test segmentation_combined
    component_cfg.return_type = 'segmentation_combined'
    ann = dataset.load_annotations(subject_dir, start_sec, duration,
        component_cfg.dict())
    
    assert tuple(ann.shape) == (duration*component_cfg.fs,)
    
    expected = np.concatenate([np.full(20, 0), np.full(10, 1), np.full(10, 2), np.full(20, 0)])
    assert ann.numpy().shape == expected.shape
    assert np.all(ann.numpy() == expected)

    # Test bbox
    component_cfg.return_type = 'bbox'
    bboxes, labels = dataset.load_annotations(subject_dir, start_sec, duration,
        component_cfg.dict())

    expected = {
        'bboxes': tf.convert_to_tensor([[20, 30], [30, 40]]),
        'labels': tf.convert_to_tensor([1, 2])
    }

    assert (bboxes.numpy() == expected['bboxes'].numpy()).all()
    assert (labels.numpy() == expected['labels'].numpy()).all()


def test_load_events_partial(ds_dir, example_config_path):
    cfg = config.parse_config(example_config_path)
    component_cfg = cfg.components['events']
    subject_dir = ds_dir / 'series1' / '10001'
    subject_dir = tf.convert_to_tensor(str(subject_dir))
    start_sec = tf.convert_to_tensor(25.0)
    duration = tf.convert_to_tensor(20.0)

    # Test segmentation_combined
    component_cfg.return_type = 'segmentation_combined'
    ann = dataset.load_annotations(subject_dir, start_sec, duration,
        component_cfg.dict())
    
    assert tuple(ann.shape) == (duration*component_cfg.fs,)
    
    expected = np.concatenate([np.full(5, 1), np.full(10, 2), np.full(5, 0)])
    assert ann.numpy().shape == expected.shape
    assert np.all(ann.numpy() == expected)

    # Test bbox
    component_cfg.return_type = 'bbox'
    bboxes, labels = dataset.load_annotations(subject_dir, start_sec, duration,
        component_cfg.dict())

    expected = {
        'bboxes': tf.convert_to_tensor([[0, 5], [5, 15]]),
        'labels': tf.convert_to_tensor([1, 2])
    }

    assert (bboxes.numpy() == expected['bboxes'].numpy()).all()
    assert (labels.numpy() == expected['labels'].numpy()).all()


def test_load_component_sample_array(ds_dir, example_config_path):
    cfg = config.parse_config(example_config_path)
    component_cfg = cfg.components['c1']
    subject_dir = ds_dir / 'series1' / '10001'
    subject_dir = tf.convert_to_tensor(str(subject_dir))
    start_sec = tf.convert_to_tensor(10.0)
    duration = 25.0
    tf_duration = tf.convert_to_tensor(duration)
    _load_func = partial(dataset.load_component, cfg=component_cfg.dict())
    s = _load_func(subject_dir, start_sec, tf_duration, duration)
    
    assert tuple(s.shape) == (duration*32, 1)
    assert s.dtype == tf.float32
    

def test_load_component_annotation(ds_dir, example_config_path):
    cfg = config.parse_config(example_config_path)
    component_cfg = cfg.components['events']
    subject_dir = ds_dir / 'series1' / '10001'
    subject_dir = tf.convert_to_tensor(str(subject_dir))
    start_sec = tf.convert_to_tensor(25.0)
    duration = 20.0
    tf_duration = tf.cast(duration, tf.float32)

    # Test segmentation_combined
    component_cfg.return_type = 'segmentation_combined'
    ann = dataset.load_component(subject_dir, start_sec, tf_duration, duration,
        component_cfg.dict())
    
    print(ann)
    assert tuple(ann.shape) == (duration*component_cfg.fs,)
    
    expected = np.concatenate([np.full(5, 1), np.full(10, 2), np.full(5, 0)])
    assert ann.numpy().shape == expected.shape
    assert np.all(ann.numpy() == expected)

    # Test bbox
    component_cfg.return_type = 'bbox'
    bboxes, labels = dataset.load_component(subject_dir, start_sec, tf_duration, duration,
        component_cfg.dict())

    expected = {
        'bboxes': tf.convert_to_tensor([[0, 5], [5, 15]]),
        'labels': tf.convert_to_tensor([1, 2])
    }

    assert (bboxes.numpy() == expected['bboxes'].numpy()).all()
    assert (labels.numpy() == expected['labels'].numpy()).all()


def test_load_element_full(ds_dir, example_config_path):
    cfg = config.parse_config(example_config_path)
    subject_dir = ds_dir / 'series1' / '10001'
    subject_dir = tf.convert_to_tensor(str(subject_dir))

    roi_start_sec = tf.convert_to_tensor(0.0)
    roi_end_sec = tf.convert_to_tensor(60.0)

    start_sec = 0.0
    duration = 60.0

    component_cfgs = {k: v.dict() for k, v in cfg.components.items()}

    elem = dataset.load_element(subject_dir,
        roi_start_sec, roi_end_sec, start_sec, duration, component_cfgs)

    assert (elem['c1'].numpy() == np.full(60*32, 0.123, dtype=np.float32)[..., np.newaxis]).all()
    assert (elem['c2'].numpy() == np.full(60*64, 1.23, dtype=np.float32)[..., np.newaxis]).all()

    assert (elem['events']['values'].numpy() == np.array([[20, 30], [30, 40]])).all()
    assert (elem['events']['labels'].numpy() == np.array([1, 2])).all()

    assert (elem['hypnogram'].numpy() == np.array([1, 0])).all()

    # Setting duration to -1.0 should give same results
    duration = -1.0
    elem = dataset.load_element(subject_dir,
        roi_start_sec, roi_end_sec, start_sec, duration, component_cfgs)

    assert (elem['c1'].numpy() == np.full(60*32, 0.123, dtype=np.float32)[..., np.newaxis]).all()
    assert (elem['c2'].numpy() == np.full(60*64, 1.23, dtype=np.float32)[..., np.newaxis]).all()

    assert (elem['events']['values'].numpy() == np.array([[20, 30], [30, 40]])).all()
    assert (elem['events']['labels'].numpy() == np.array([1, 2])).all()

    assert (elem['hypnogram'].numpy() == np.array([1, 0])).all()


def test_load_element_partial(ds_dir, example_config_path):
    cfg = config.parse_config(example_config_path)
    subject_dir = ds_dir / 'series1' / '10001'
    subject_dir = tf.convert_to_tensor(str(subject_dir))

    roi_start_sec = tf.convert_to_tensor(0.0)
    roi_end_sec = tf.convert_to_tensor(60.0)

    start_sec = 30.0
    duration = 30.0

    component_cfgs = {k: v.dict() for k, v in cfg.components.items()}

    elem = dataset.load_element(subject_dir,
        roi_start_sec, roi_end_sec, start_sec, duration, component_cfgs)

    assert (elem['c1'].numpy() == np.full(30*32, 0.123, dtype=np.float32)[..., np.newaxis]).all()
    assert (elem['c2'].numpy() == np.full(30*64, 1.23, dtype=np.float32)[..., np.newaxis]).all()

    assert (elem['events']['values'].numpy() == np.array([[0, 10]])).all()
    assert (elem['events']['labels'].numpy() == np.array([2])).all()

    assert (elem['hypnogram'].numpy() == np.array([0])).all()


def test_load_element_random_start_sec(ds_dir, example_config_path):
    cfg = config.parse_config(example_config_path)
    subject_dir = ds_dir / 'series1' / '10001'
    subject_dir = tf.convert_to_tensor(str(subject_dir))

    roi_start_sec = tf.convert_to_tensor(0.0)
    roi_end_sec = tf.convert_to_tensor(60.0)

    start_sec = -1.0
    duration = 30.0

    component_cfgs = {k: v.dict() for k, v in cfg.components.items()}

    elem = dataset.load_element(subject_dir,
        roi_start_sec, roi_end_sec, start_sec, duration, component_cfgs)

    assert (elem['c1'].numpy() == np.full(30*32, 0.123, dtype=np.float32)[..., np.newaxis]).all()
    assert (elem['c2'].numpy() == np.full(30*64, 1.23, dtype=np.float32)[..., np.newaxis]).all()

    assert elem['hypnogram'].numpy().shape == (1,)


def test_load_dataset(ds_dir, example_config_path):
    cfg = config.parse_config(example_config_path)
    cfg.ds_dir = ds_dir
    slf_ds = slf.reader.read_dataset(ds_dir)
    tf_ds = dataset.from_slf_dataset(slf_ds, cfg)
    
    numpy_iter = tf_ds.as_numpy_iterator()
    for elem in numpy_iter:
        print(elem)

    # Should raise StopIteration because whole dataset was already iterated
    with pytest.raises(StopIteration):
        numpy_iter.next()
    