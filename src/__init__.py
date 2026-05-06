import logging
import sys
import pathlib

pathlib.Path("logs").mkdir(exist_ok=True)
logpath = pathlib.Path.joinpath("logs", "penguin_mlops.log").touch(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(logpath)]
)
logger = logging.getLogger(__name__)