import sys
print("Python executable:", sys.executable)
print("Module search path:", sys.path)

from hydra import main
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch_points3d.trainer import Trainer
import logging

@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    
    numba_logger = logging.getLogger('numba')
    numba_logger.setLevel(logging.WARNING)
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    if cfg.pretty_print:
        print(OmegaConf.to_yaml(cfg))

    trainer = Trainer(cfg)
    print("NUM_CLASSES =", cfg.data.num_classes)
    trainer.train()
    #
    # # https://github.com/facebookresearch/hydra/issues/440
    GlobalHydra.get_state().clear()
    return 0


if __name__ == "__main__":
    main()
