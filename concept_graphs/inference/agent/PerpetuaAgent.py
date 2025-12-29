from typing import Dict, Any, Tuple, Union

import jax.numpy as jnp

from langchain.agents import create_agent
from langchain.tools import tool

from semistaticsim.groundtruth.simulator import Simulator

from concept_graphs.inference.agent.BaseAgent import Agent, AgentPrinterCallback
from concept_graphs.inference.toolbox.PerpetuaMapToolbox import PerpetuaMapToolbox
from concept_graphs.mapping.PerpetuaObjectMap import PerpetuaObjectMap

class PerpetuaAgent(Agent):
    def __init__(self, sim: Simulator, toolbox: PerpetuaMapToolbox):
        super().__init__(sim)
        self.toolbox = toolbox
        self.llm_model = "openai:gpt-5"
        # self.llm_model = "gpt-4.1"
        self.llm_agent = create_agent(model=self.llm_model, tools=self.tools())


    @property
    def object_map(self) -> PerpetuaObjectMap:
        return self.toolbox.object_map

    def tools(self):
        @tool
        def predict_object_receptacle(pickupable_id) -> Tuple[str, Dict[str, float]]:
            """

            Predicts the current receptacle for an object.
            The results of this tool should be trusted above any semantic prior you might have!
            If the most likely receptacle reported by this tool seems nonsensical, trust it anyway. It has learned from real data.

            Args:
                pickupable_id: A pickupable object's id. Must match exactly.

            Returns:
                argmax_receptacle (str): Most likely receptacle
                recptacle_weights (Dict[str, float]) Dict of probability weights for each receptacle]

            """
            return self.predict_object_receptacle(pickupable_id)

        @tool
        def go_to_receptacle(receptacle_id: str, pickupable_id: str):
            """

            Args:
                receptacle_id: The name of the target receptacle.
                looking_for_pickupable: The name of the pickupable we are looking for.

            Returns: True if the current state contains the pickupable, False otherwise.

            """
            return self.go_to_receptacle(
                receptacle_id, pickupable_id
            )  # .go_to_receptacle(receptacle_id, pickupable_id)

        return [predict_object_receptacle, go_to_receptacle]

    def query(self, query: str):
        system_message = {
            "role": "system",
            "content": f"""
You are an embodied LLM planner. 

Here are the IDs of the objects of interest: <{', '.join(self.sim.sss_data.pickupable_names)}>
Each object of interest will be in different <receptacles> through time.
The receptacles IDs are: <{', '.join(self.sim.sss_data.receptacle_names)}>

You cannot assume the current receptacle of any object of interest. 
You must use your prediction tool to figure out where the object currently is.

You must reason about the user query; some only require prediction, some only require navigation, some require both.
Do not go to an object unless explicitly necessary to fulfill the query.
""",
        }

        user_message = {"role": "user", "content": query}

        return self.llm_agent.invoke(
            {"messages": [system_message, user_message]},
            config={"callbacks": [AgentPrinterCallback()]},
        )

    @property
    def current_time(self):
        return float(self.sim.sss_data.self_at_current_time._timestamp)

    def _predict_object_receptacle(
        self, pickupable_id: str, current_time: float = None
    ) -> Dict[str, float]:
        """

        Args:
            pickupable_id: A pickupable object's id. Must match exactly.

        Returns: Tuple[Most likely receptacle, Dict of probability weights for each receptacle]

        """
        pickupable_name = self.resolve_query_into_pickupable(pickupable_id)
        prediction, _ = self.toolbox.temporal_object_query(pickupable_name, current_time)
        sorted_keys = sorted(prediction, key=prediction.get, reverse=True)
        return sorted_keys[0], prediction

    def update(self, observation: Dict[str, Any]):
        apn = observation["point_apn"]
        assignments = apn._assignment
        object_containment_obs = {}
        for pickupable_idx, pickupable_id in enumerate(apn.pickupable_names):
            pickupable_vector = assignments[pickupable_idx]
            # Check if there is a positive observation
            found_positive = (pickupable_vector == 1.0).any()
            if found_positive:
                obs_receptacle = {}
                # If a pickupable is seen in a receptacle, we can assume then it is not in other receptacles
                for idx, val in enumerate(pickupable_vector):
                    name = apn.receptacle_names[idx]
                    if name == 'OOB_FAKE_RECEPTACLE':
                        continue
                    if val == 1.0:
                        obs_receptacle[name] = jnp.array([1.0])
                    elif val >= -1.0:
                        obs_receptacle[name] = jnp.array([0.0])
            else:
                # If not found, just return the places where it was not seen (if they exist)
                obs_receptacle = {
                    apn.receptacle_names[idx]: val
                    for idx, val in enumerate(pickupable_vector)
                    if val >= 0 and apn.receptacle_names[idx] != 'OOB_FAKE_RECEPTACLE'
                }
            # Only populate if we have some observation
            if len(obs_receptacle) > 0:
                object_containment_obs[pickupable_id] = obs_receptacle
        self.toolbox.temporal_map_update(object_containment_obs, self.current_time)


    def found_pickupable(
        self, receptacle_id: str, pickupable_id: str, observation: Dict[str, Any]
    ) -> Union[bool, None]:
        """

        Args:
            observation: An observation of the current state.

        Returns: True if the current state contains the pickupable. False if the receptacle doesn't have the pickupable. None if no information was gained.

        """
        from semistaticsim.datawrangling.sssd import GeneratedSemiStaticData

        apn: GeneratedSemiStaticData = observation["point_apn"]

        vector_for_pickupable = apn._assignment[
            apn.pickupable_names.index(pickupable_id)
        ]
        if (vector_for_pickupable == 1).any():
            return True

        try:
            if vector_for_pickupable[apn.receptacle_names.index(receptacle_id)] >= 0:
                return False
        except:
            pass

        return None