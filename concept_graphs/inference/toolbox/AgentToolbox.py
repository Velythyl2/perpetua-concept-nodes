import logging

from semistaticsim.groundtruth.simulator import Simulator

from concept_graphs.inference.toolbox.PerpetuaMapToolbox import PerpetuaMapToolbox
from concept_graphs.inference.agent.PerpetuaAgent import PerpetuaAgent

log = logging.getLogger(__name__)


class AgentToolbox(PerpetuaMapToolbox):
    def __init__(self, sim: Simulator, **kwargs):
        super().__init__(**kwargs)
        self.agent = PerpetuaAgent(sim, toolbox=self)

    def open_vocab_query(self, user_query: str):
        response = self.agent.query(user_query)
        return response

    