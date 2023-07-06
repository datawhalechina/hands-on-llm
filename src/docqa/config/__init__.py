import logging
import os
from os.path import dirname, abspath, join

import pnlp


class Config(pnlp.MagicDict):

    def __getitem__(self, key: str):
        if key not in self:
            msg = f"key: {key} not in config"
            raise KeyError(msg)
        return self.get(key)


ROOT = dirname(abspath(__file__))

LOG_ROOT = join(dirname(ROOT), "logs")
pnlp.check_dir(LOG_ROOT)

logging.basicConfig(filename=join(LOG_ROOT, "docqa.log"), level=logging.INFO)
logger = logging.getLogger(__file__)


STREAM_DELAY = 0.1

STREAM_RETRY_TIMEOUT = 30000  # NOTE: used for sse
TRITON_CLIENT_TIMEOUT = 3
TRITON_RETRY_TIMES = 3

FUSE_COUNT = 10

cfg = pnlp.read_yaml(join(ROOT, "config.yaml"))


llm_config = Config(cfg["llm"])
emd_config = Config(cfg["emb"])
recaller_config = Config(cfg["recall"])


PROFILE = os.environ.get("PROFILE", "dev")
