from typing import Union
import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import logging

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
jax.config.update("jax_platform_name", "cpu")

from concept_graphs.utils import set_seed

from concept_graphs.inference.agent.PerpetuaAgent import PerpetuaAgent
from concept_graphs.viz.server.PerpetuaMapServer import PerpetuaMapServer

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="agent")
def main(cfg: DictConfig):
    # If running (procthor) agent, instantiate simulator!
    from semistaticsim.keyboardcontrol import main_skillsim
    from pathlib import Path

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    cfg_path = Path(cfg.data_dir) / cfg.scene
    gt_cfg_path = Path(cfg.data_dir).parent / "groundtruth"
    start_at = f"{cfg.scene.split('_')[1]} hour"

    sim = main_skillsim.main(
        config_path=str(cfg_path),
        overrides=[
            "mode.runfunc=rollout",
            "mode.get_simulator_instance=true",
            f"scene={str(gt_cfg_path)}",
            f"start_at={start_at}",
        ],
    )

    set_seed(cfg.seed)
    # server: PerpetuaMapServer = instantiate(cfg.server)
    agent: PerpetuaAgent = instantiate(cfg.agent, sim=sim)
    # Start and spin the server
    # server.toolbox.agent.go_to_receptacle('Dresser|4|1', 'AlarmClock|surface|4|12')
    agent.spin()

    """
    Note: the loop below is meant for open-vocab semistatic llm planning.
    For closed-vocab, use the existing functions like so:
    1. get target pickupable ID
    2. target_receptacle, _ = agent.predict_object_receptacle(pickupable_id)
    3. agent.go_to_receptacle(target_receptacle, pickupable_id)
    """


if __name__ == "__main__":
    main()
