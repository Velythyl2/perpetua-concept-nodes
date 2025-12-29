"""

EVENTUALLY, THIS FILE WILL GO INTO SSS REPO

"""

from abc import ABC
from typing import Dict, Any, Tuple, Optional

from semistaticsim.keyboardcontrol.main_skillsim import ROBOTS
from semistaticsim.rendering.simulation.skill_simulator import Simulator

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class Agent(ABC):
    def __init__(self, sim: Simulator):
        self.sim = sim
        # TODO: make these parameters configurable
        self.dt = 1 / 1000    # Assume a full loop takes less than 1000 steps
        self.iterations = 0   # To track the number of updates

    @property
    def current_time(self):
        return self.sim.sss_data.pickupable_selves_at_current_time._timestamp[0] + self.dt * self.iterations

    def resolve_query_into_pickupable(self, query: str) -> str:
        # assumes query is a cleaned up name
        for p in self.sim.sss_data.pickupable_names:
            if p.startswith(query):
                return p

    def _predict_object_receptacle(
        self, pickupable_id: str, current_time: float = None
    ) -> Tuple[str, Dict[str, float]]:
        """

        Args:
            pickupable_id: A requested pickupable.
            current_time:

        Returns: A probability weight for each receptacle for the current location of the pickupable.

        """
        raise NotImplementedError()

    def predict_object_receptacle(
        self, pickupable_id: str, current_time: float = None
    ) -> Tuple[str, Dict[str, float]]:
        """

        Args:
            pickupable_id: A pickupable object's id. Must match exactly.

        Returns: Tuple[Most likely receptacle, Dict of probability weights for each receptacle]

        """

        if current_time is None:
            current_time = self.current_time

        return self._predict_object_receptacle(pickupable_id, current_time)

    def update(self, observation: Dict[str, Any]) -> None:
        """
        Updates self given the observation of the environment

        Args:
            observation: An observation of the current state.

        Returns:

        """
        raise NotImplementedError()

    def found_pickupable(
        self, receptacle_id: str, pickupable_id: str, observation: Dict[str, Any]
    ) -> bool:
        """

        Args:
            observation: An observation of the current state.

        Returns: True if the current state contains the pickupable, False otherwise.

        """
        raise NotImplementedError()

    def go_to_receptacle(
        self, receptacle_id: str, looking_for_pickupable: str
    ) -> Tuple[Dict, bool]:
        """

        Args:
            receptacle_id: The id of the target receptacle.
            looking_for_pickupable: The id of the pickupable we are looking for.

        Returns: True if the current state contains the pickupable, False otherwise.

        """
        while True:
            is_pathing_finished = self.sim.GoToObject(
                ROBOTS[0], receptacle_id, max_path_length=2
            )
            obs = self.sim.render()
            self.sim.privileged_apn = None

            self.update(obs)
            self.iterations += 1

            found_it = self.found_pickupable(
                receptacle_id, looking_for_pickupable, obs
            )
            if found_it is not None:
                return obs, found_it
            if is_pathing_finished:
                return obs, False

    def goto_query(
        self,
        weighted_receptacles=None,
        pickupable_id=None,
        query: str = None,
        current_time: float = None,
        custom_traversal=None,
    ) -> Tuple[Dict, bool]:
        if pickupable_id is None:
            assert query is not None

            pickupables = self.resolve_query_into_pickupable(query, current_time)
            pickupable_id = pickupables[0]

        if weighted_receptacles is None:
            weighted_receptacles = self.resolve_pickupable_into_weighted_receptacles(
                pickupable_id, current_time
            )

        sorted_keys = sorted(
            weighted_receptacles, key=weighted_receptacles.get, reverse=True
        )

        if custom_traversal is not None:
            return custom_traversal(sorted_keys)

        for k in sorted_keys:
            last_obs, found_obj = self.go_to_receptacle(k, pickupable_id)
            if found_obj:
                return last_obs, True
        return last_obs, False


class AgentPrinterCallback(BaseCallbackHandler):
    """
    Custom callback to print LLM reasoning and tool usage to the console.
    """

    def on_chat_model_start(self, serialized, messages, **kwargs):
        # Optional: Print "Thinking..." if you want to know it started
        pass

    def on_llm_end(self, response: LLMResult, **kwargs):
        """Print the LLM's reasoning or text response."""
        # Access the actual text generated
        if response.generations and response.generations[0]:
            generation = response.generations[0][0]
            # Check if it's a ChatGeneration (standard for OpenAI)
            if hasattr(generation, "message"):
                content = generation.message.content
                # Only print if there is actual text (sometimes it's just a tool call)
                if content:
                    print(f"\n\033[1m[LLM Reasoning]\033[0m:\n{content}\n")
            # Fallback for standard generation
            elif generation.text:
                print(f"\n\033[1m[LLM Reasoning]\033[0m:\n{generation.text}\n")

    def on_tool_start(self, serialized, input_str, **kwargs):
        """Print which tool is being called and with what inputs."""
        tool_name = serialized.get("name")
        print(f"\033[94m[Tool Call]\033[0m: {tool_name}")
        print(f"\033[94m[Arguments]\033[0m: {input_str}")

    def on_tool_end(self, output, **kwargs):
        """Print the result from the tool."""
        print(f"\033[92m[Tool Output]\033[0m: {str(output)}\n")


"""

EVENTUALLY, THIS FILE WILL GO INTO SSS REPO

"""
