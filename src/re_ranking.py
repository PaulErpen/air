from .pipeline.pipeline import Pipeline
from .config.config import create_base_config

cfg = create_base_config()
#cfg.model = "tk"
pipeline = Pipeline.from_config(cfg)
pipeline.evaluate()
