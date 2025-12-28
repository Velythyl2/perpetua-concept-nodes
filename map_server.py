from typing import Union
import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import logging
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
# Disable GPU memory pre-allocation to avoid OOM
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax

jax.config.update("jax_platform_name", "cpu")
import jax

from concept_graphs.utils import set_seed


from concept_graphs.viz.server.ObjectMapServer import ObjectMapServer
from concept_graphs.viz.server.PerpetuaMapServer import PerpetuaMapServer

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="map_server")
def main(cfg: DictConfig):
    server_target_class = cfg.server._target_
    sim = None
    # If running (procthor) agent, instantiate simulator!
    if "AgentServer" in server_target_class:
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
    toolbox = instantiate(cfg.server.toolbox, sim=sim) # Pass sim here as otherwise it does not get passed to AgentToolbox
    server: Union[ObjectMapServer | PerpetuaMapServer] = instantiate(cfg.server, toolbox=toolbox)
    log.info(f"Loading map with a total of {len(server.object_map)} objects")
    # Start and spin the server
    server.spin()

    """
    Note: the loop below is meant for open-vocab semistatic llm planning.
    For closed-vocab, use the existing functions like so:
    1. get target pickupable ID
    2. target_receptacle, _ = agent.predict_object_receptacle(pickupable_id)
    3. agent.go_to_receptacle(target_receptacle, pickupable_id)
    """


if __name__ == "__main__":
    main()
