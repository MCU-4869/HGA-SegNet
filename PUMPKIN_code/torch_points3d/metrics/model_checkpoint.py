from typing import Any, Dict, List, Optional, Tuple, Union
import os
import torch
import logging
import copy
import glob
import shutil
from omegaconf import OmegaConf
from omegaconf import DictConfig

from torch_points3d.models import model_interface
from torch_points3d.utils.colors import COLORS, colored_print
from torch_points3d.core.schedulers.lr_schedulers import instantiate_scheduler
from torch_points3d.core.schedulers.bn_schedulers import instantiate_bn_scheduler
from torch_points3d.models.model_factory import instantiate_model

log = logging.getLogger(__name__)


def _alias_numpy_core():
    """Make checkpoints saved under numpy>=2 loadable on numpy 1.x.

    numpy 2 renamed `numpy.core` to `numpy._core`, and the pickle inside a
    `torch.save`d checkpoint stores the *module path* of every ndarray. A file
    written under numpy 2 therefore references `numpy._core.multiarray`, which
    simply does not exist in a numpy 1.x environment (this project runs numpy
    1.x on python 3.8) -> `ModuleNotFoundError: No module named 'numpy._core'`
    raised from inside `torch.load`, before any of our code sees the object.

    2026-11-19: the slimmed checkpoint packs are written on a Windows box with
    numpy 2, so every one of them failed to load on the cloud until this alias
    was added. It is a pure shim: the array layouts are identical, only the
    module name changed. On numpy>=2 `hasattr(numpy, "_core")` is True and this
    returns immediately, so nothing changes there.
    """
    import sys as _sys

    import numpy as _np

    if hasattr(_np, "_core"):
        return
    try:
        import numpy.core as _core
    except Exception:                                        # pragma: no cover
        return
    _sys.modules.setdefault("numpy._core", _core)
    for _name in dir(_core):
        _mod = getattr(_core, _name, None)
        if getattr(_mod, "__name__", "").startswith("numpy.core"):
            _sys.modules.setdefault("numpy._core." + _name, _mod)


_alias_numpy_core()


class Checkpoint:
    _LATEST = "latest"

    def __init__(self, checkpoint_file: str, save_every_iter: bool = True):
        """ Checkpoint manager. Saves to working directory with check_name
        Arguments
            checkpoint_file {str} -- Path to the checkpoint
            save_every_iter {bool} -- [description] (default: {True})
        """
        self._check_path = checkpoint_file
        self._filled = False
        self.run_config: Optional[Dict] = None
        self.models: Dict[str, Any] = {}
        self.stats: Dict[str, List[Any]] = {"train": [], "test": [], "val": []}
        self.optimizer: Optional[Tuple[str, Any]] = None
        self.schedulers: Dict[str, Any] = {}
        self.dataset_properties: Dict = {}

    def save_objects(self, models_to_save: Dict[str, Any], stage, current_stat, optimizer, schedulers, **kwargs):
        """ Saves checkpoint with updated mdoels for the given stage
        """
        self.models = models_to_save
        self.optimizer = (optimizer.__class__.__name__, optimizer.state_dict())
        self.schedulers = {
            scheduler_name: [scheduler.scheduler_opt, scheduler.state_dict()]
            for scheduler_name, scheduler in schedulers.items()
        }
        to_save = kwargs
        for key, value in self.__dict__.items():
            if not key.startswith("_"):

                to_save[key] = value
        torch.save(to_save, self.path)

    @property
    def path(self):
        return self._check_path

    @staticmethod
    def load(checkpoint_dir: str, checkpoint_name: str, run_config: Any, strict=False, resume=True):
        """ Creates a new checkpoint object in the current working directory by loading the
        checkpoint located at [checkpointdir]/[checkpoint_name].pt
        """
        checkpoint_file = os.path.join(checkpoint_dir, checkpoint_name) + ".pt"
        if not os.path.exists(checkpoint_file):
            ckp = Checkpoint(checkpoint_file)
            if strict or resume:
                available_checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
                message = "The provided path {} didn't contain the checkpoint_file {}".format(
                    checkpoint_dir, checkpoint_name + ".pt"
                )
                if available_checkpoints:
                    message += "\nDid you mean {}?".format(os.path.basename(available_checkpoints[0]))
                raise ValueError(message)
            ckp.run_config = run_config
            return ckp
        else:
            chkp_name = os.path.basename(checkpoint_file)
            if resume:
                shutil.copyfile(
                    checkpoint_file, chkp_name
                )  # Copy checkpoint to new run directory to make sure we don't override
            ckp = Checkpoint(chkp_name)
            #ckp.run_config = run_config
            log.info("Loading checkpoint from {}".format(checkpoint_file))
            objects = torch.load(checkpoint_file, map_location="cpu")
            for key, value in objects.items():
                setattr(ckp, key, value)
                
            #during training, uncomment the following line
            #ckp.run_config['models']['PointGroup-PAPER']['cluster_type'] = run_config['models']['PointGroup-PAPER']['cluster_type']
            #ckp.run_config['epochs'] = run_config['epochs']
            #ckp.run_config['training']['epochs'] = run_config['training']['epochs']
            ckp._filled = True
        return ckp

    @property
    def is_empty(self):
        return not self._filled

    def load_optim_sched(self, model, load_state=True):
        if not self.is_empty:
            # ★ 2026-11-19: slim (weights-only) checkpoints carry no optimizer
            # state. Rendering / evaluation never uses it — the Trainer builds a
            # fresh optimizer right after this call. Training checkpoints always
            # carry it, so their behaviour is unchanged.
            optimizer_config = getattr(self, "optimizer", None)
            if optimizer_config is None:
                log.info("Checkpoint carries no optimizer state (slim file) — "
                         "skipping optimizer/scheduler restore")
                return {}
            # initialize optimizer
            optimizer_cls = getattr(torch.optim, optimizer_config[0])
            optimizer_params = OmegaConf.create(self.run_config).training.optim.optimizer.params
            model.optimizer = optimizer_cls(model.parameters(), **optimizer_params)

            # initialize & load schedulersr
            schedulers_out = {}
            schedulers_config = self.schedulers
            for scheduler_type, (scheduler_opt, scheduler_state) in schedulers_config.items():
                if scheduler_type == "lr_scheduler":
                    optimizer = model.optimizer
                    scheduler = instantiate_scheduler(optimizer, OmegaConf.create(scheduler_opt))
                    if load_state:
                        scheduler.load_state_dict(scheduler_state)
                    schedulers_out["lr_scheduler"] = scheduler

                elif scheduler_type == "bn_scheduler":
                    scheduler = instantiate_bn_scheduler(model, OmegaConf.create(scheduler_opt))
                    if load_state:
                        scheduler.load_state_dict(scheduler_state)
                    schedulers_out["bn_scheduler"] = scheduler

            # load optimizer
            model.schedulers = schedulers_out
            if load_state:
                model.optimizer.load_state_dict(optimizer_config[1])

    def get_state_dict(self, weight_name):
        if not self.is_empty:
            try:
                models = self.models
                keys = [key.replace("best_", "") for key in models.keys()]
                log.info("Available weights : {}".format(keys))
                try:
                    key_name = "best_{}".format(weight_name)
                    model = models[key_name]
                    log.info("Model loaded from {}:{}.".format(self._check_path, key_name))
                    return model
                except:
                    key_name = Checkpoint._LATEST
                    model = models[Checkpoint._LATEST]
                    log.info("Model loaded from {}:{}".format(self._check_path, key_name))
                    return model
            except:
                raise Exception("This weight name isn't within the checkpoint ")


class ModelCheckpoint(object):
    """ Create a checkpoint for a given model

    Argumemnts:
        - load_dir: directory where to load the checkpoint from (if exists)
        - check_name: Name of the checkpoint (without the .pt extension)
        - selection_stage: Stage that is used for selecting the best model
        - run_config: Config of the run. In resume mode, this gets discarded
        - resume: Resume a previous training - this creates optimizers
        - strict: If strict and checkpoint is empty then it raises a ValueError. Being in resume mode forces strict
    """

    def __init__(
        self,
        load_dir: str,
        check_name: str,
        selection_stage: str,
        run_config: DictConfig = DictConfig({}),
        resume=False,
        strict=False,
    ):
        # Conversion of run_config to save a dictionary and not a pickle of omegaconf
        rc = OmegaConf.to_container(copy.deepcopy(run_config))
        self._checkpoint = Checkpoint.load(load_dir, check_name, run_config=rc, strict=strict, resume=resume)
        # `fold` is a PUMPKIN-only key (it selects the test area). The target-domain
        # data configs (crops3d_is / icrisat / bonnbeet) do not define it, and a
        # *fresh training run* used to die here with ConfigAttributeError before a
        # single batch ran. Two cases, kept distinct on purpose:
        #   * the live config HAS fold  -> keep the historical overwrite. This is
        #     what lets eval_zeroshot.py inject a fold into a stored PUMPKIN config.
        #   * the live config has NO fold -> do not touch the stored value if it has
        #     one; otherwise default to [] so the rest of the code can read it.
        _live_fold = run_config.data.get("fold", None)
        if _live_fold is not None:
            self._checkpoint.run_config['data']['fold'] = _live_fold
        elif 'fold' not in self._checkpoint.run_config.get('data', {}):
            self._checkpoint.run_config.setdefault('data', {})['fold'] = []
        # Same mechanism for the frozen-sample paths (2026-11-18). Why it is needed:
        # eval rebuilds the dataset from the SAVED config, so a CLI
        # `data.frozen_test_samples=...` / `+data.frozen_test_samples=...` would be
        # silently ignored -- and the stored value may point at another machine's
        # absolute path, which then raises FileNotFoundError in
        # `_apply_frozen_samples` (it checks the path before looking at whether the
        # split exists, so even a missing *val* file kills a test-only eval).
        # Only an explicit live value overwrites; an absent live key leaves the
        # stored one untouched, exactly like `fold` above.
        for _key in ("frozen_test_samples", "frozen_val_samples"):
            _live = run_config.data.get(_key, None)
            if _live is not None:
                self._checkpoint.run_config.setdefault('data', {})[_key] = _live
        self._resume = resume
        self._selection_stage = selection_stage

    def create_model(self, dataset, weight_name=Checkpoint._LATEST):
        if not self.is_empty:
            run_config = copy.deepcopy(self._checkpoint.run_config)
            model = instantiate_model(OmegaConf.create(run_config), dataset)
            if hasattr(self._checkpoint, "model_props"):
                for k, v in self._checkpoint.model_props.items():
                    setattr(model, k, v)
                delattr(self._checkpoint, "model_props")
            self._initialize_model(model, weight_name)
            return model
        else:
            raise ValueError("Checkpoint is empty")

    @property
    def start_epoch(self):
        if self._resume:
            return self.get_starting_epoch()
        else:
            return 1

    @property
    def run_config(self):
        return OmegaConf.create(self._checkpoint.run_config)

    @property
    def data_config(self):
        return OmegaConf.create(self._checkpoint.run_config).data

    @property
    def selection_stage(self):
        return self._selection_stage

    @selection_stage.setter
    def selection_stage(self, value):
        self._selection_stage = value

    @property
    def is_empty(self):
        return self._checkpoint.is_empty

    @property
    def checkpoint_path(self):
        return self._checkpoint.path

    @property
    def dataset_properties(self) -> Dict:
        return self._checkpoint.dataset_properties

    @dataset_properties.setter
    def dataset_properties(self, dataset_properties: Union[Dict[str, Any], Dict]):
        self._checkpoint.dataset_properties = dataset_properties

    def get_starting_epoch(self):
        return len(self._checkpoint.stats["train"]) + 1

    def _initialize_model(self, model: model_interface.CheckpointInterface, weight_name):
        if not self._checkpoint.is_empty:
            state_dict = self._checkpoint.get_state_dict(weight_name)
            model.load_state_dict(state_dict, strict=False)
            self._checkpoint.load_optim_sched(model, load_state=self._resume)

    def restore_optim_sched(self, model: model_interface.CheckpointInterface):
        """Re-apply the checkpoint's optimizer / LR-scheduler / BN-scheduler states.

        Public re-entry to the inner Checkpoint's load_optim_sched. The trainer needs
        this on the resume path: it must call model.instantiate_optimizers() (the only
        place that sets _grad_scale / _grad_clip / _accumulated_gradient_step) AFTER
        create_model() has already restored the training state, and that call builds a
        fresh optimizer which clobbers the restored one. load_optim_sched rebuilds the
        optimizer from the checkpoint's state_dict, so calling it again is idempotent
        and puts back exactly what was clobbered.
        """
        self._checkpoint.load_optim_sched(model, load_state=True)
    def find_func_from_metric_name(self, metric_name, default_metrics_func):
        for token_name, func in default_metrics_func.items():
            if token_name in metric_name:
                return func
        raise Exception(
            'The metric name {} doesn t have a func to measure which one is best in {}. Example: For best_train_iou, {{"iou":max}}'.format(
                metric_name, default_metrics_func
            )
        )

    def save_best_models_under_current_metrics(
        self, model: model_interface.CheckpointInterface, metrics_holder: dict, metric_func_dict: dict, **kwargs
    ):
        """[This function is responsible to save checkpoint under the current metrics and their associated DEFAULT_METRICS_FUNC]
        Arguments:
            model {[CheckpointInterface]} -- [Model]
            metrics_holder {[Dict]} -- [Need to contain stage, epoch, current_metrics]
        """
        metrics = metrics_holder["current_metrics"]
        stage = metrics_holder["stage"]
        epoch = metrics_holder["epoch"]

        stats = self._checkpoint.stats
        state_dict = copy.deepcopy(model.state_dict())

        current_stat = {}
        current_stat["epoch"] = epoch

        models_to_save = self._checkpoint.models
        if stage not in stats:
            stats[stage] = []

        if stage == "train":
            models_to_save[Checkpoint._LATEST] = state_dict
        else:
            if len(stats[stage]) > 0:
                latest_stats = stats[stage][-1]

                msg = ""
                improved_metric = 0

                for metric_name, current_metric_value in metrics.items():
                    current_stat[metric_name] = current_metric_value

                    metric_func = self.find_func_from_metric_name(metric_name, metric_func_dict)
                    best_metric_from_stats = latest_stats.get("best_{}".format(metric_name), current_metric_value)
                    best_value = metric_func(best_metric_from_stats, current_metric_value)
                    current_stat["best_{}".format(metric_name)] = best_value

                    # This new value seems to be better under metric_func
                    if (self._selection_stage == stage) and (
                        current_metric_value == best_value
                    ):  # Update the model weights
                        models_to_save["best_{}".format(metric_name)] = state_dict

                        msg += "{}: {} -> {}, ".format(metric_name, best_metric_from_stats, best_value)
                        improved_metric += 1

                if improved_metric > 0:
                    colored_print(COLORS.VAL_COLOR, msg[:-2])
            else:
                # stats[stage] is empty.
                for metric_name, metric_value in metrics.items():
                    current_stat[metric_name] = metric_value
                    current_stat["best_{}".format(metric_name)] = metric_value
                    models_to_save["best_{}".format(metric_name)] = state_dict

        kwargs["model_props"] = {
            "num_epochs": model.num_epochs,  # type: ignore
            "num_batches": model.num_batches,  # type: ignore
            "num_samples": model.num_samples,  # type: ignore
        }

        self._checkpoint.stats[stage].append(current_stat)
        self._checkpoint.save_objects(models_to_save, stage, current_stat, model.optimizer, model.schedulers, **kwargs)

    def validate(self, dataset_config):
        """ A checkpoint is considered as valid if it can recreate the model from
        a dataset config only """
        if dataset_config is not None:
            for k, v in dataset_config.items():
                self.data_config[k] = v
        try:
            instantiate_model(OmegaConf.create(self.run_config), self.data_config)
        except:
            return False

        return True
