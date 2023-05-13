from .train_handler.train_handler import TrainHandler
from .config.config import create_base_config

cfg = create_base_config()
handler = TrainHandler.from_config(cfg)
handler.train()
