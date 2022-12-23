import yaml

from pathlib import Path
from pydantic import BaseModel, Extra


class ComponentConfig(BaseModel, extra=Extra.forbid):
    src_name: str
    ctype: str
    fs: float | None
    sampling_interval: float | None
    value_map: dict[str, int] | None
    return_type: str | None


class DatasetConfig(BaseModel, extra=Extra.forbid):
    ds_dir: Path
    series_name: str
    components: dict[str, ComponentConfig]

    start_sec: float
    duration: float
    roi_src_type: str
    roi_src_name: str


def parse_config(config_path: Path) -> DatasetConfig:
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    return DatasetConfig.parse_obj(cfg)