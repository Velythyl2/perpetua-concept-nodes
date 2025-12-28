from typing import Dict, Any, Tuple, Union

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
        self, pickupable: str, current_time: float = None
    ) -> Dict[str, float]:
        """

        Args:
            pickupable_id: A pickupable object's id. Must match exactly.

        Returns: Tuple[Most likely receptacle, Dict of probability weights for each receptacle]

        """
        pickupable_id = self.resolve_query_into_pickupable(pickupable)
        prediction = self.object_map.object_predict(pickupable_id, current_time)
        sorted_keys = sorted(prediction, key=prediction.get, reverse=True)
        return sorted_keys[0], prediction

    def update(self, observation: Dict[str, Any]):
        return None  # todo self.object_map.object_update(current_time, pickupable)

    def found_pickupable(
        self, receptacle_name: str, pickupable_name: str, observation: Dict[str, Any]
    ) -> Union[bool, None]:
        """

        Args:
            observation: An observation of the current state.

        Returns: True if the current state contains the pickupable. False if the receptacle doesn't have the pickupable. None if no information was gained.

        """
        from semistaticsim.datawrangling.sssd import GeneratedSemiStaticData

        apn: GeneratedSemiStaticData = observation["point_apn"]

        vector_for_pickupable = apn._assignment[
            apn.pickupable_names.index(pickupable_name)
        ]
        if (vector_for_pickupable == 1).any():
            return True

        try:
            if vector_for_pickupable[apn.receptacle_names.index(receptacle_name)] >= 0:
                return False
        except:
            pass

        return None