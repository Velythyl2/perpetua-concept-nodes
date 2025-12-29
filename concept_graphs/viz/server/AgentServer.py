from typing import List, Optional, Dict
from scipy.spatial.transform import Rotation as Rsc
import numpy as np
import viser

from concept_graphs.inference.toolbox.PerpetuaMapToolbox import PerpetuaMapToolbox
from concept_graphs.inference.toolbox.AgentToolbox import AgentToolbox
from concept_graphs.viz.server.PerpetuaMapServer import PerpetuaMapServer

import logging

log = logging.getLogger(__name__)


class AgentServer(PerpetuaMapServer):
    def __init__(self, toolbox: AgentToolbox, point_shape: str = "circle"):
        super().__init__(toolbox, point_shape)

        self.open_vocab_query = None  

        # GUI
        with self.tab_group.add_tab("Agent"):
            with self.server.gui.add_folder("Open-Vocab"):
                self.open_vocab_query_gui_text = self.server.gui.add_text(
                    "Query",
                    initial_value="",
                )
                self.open_vocab_query_button = self.server.gui.add_button(
                        "Submit Query",
                        icon=viser.Icon.MOUSE,
                    )
                self.open_vocab_query_button.on_click(self.on_open_vocab_query_submit)

    # Gui callbacks and inputs
    def on_open_vocab_query_submit(self, data):
        self.open_vocab_query = self.open_vocab_query_gui_text.value

    # Register here all callbacks that are resource intenseful and need to be called in the main loop
    def _callbacks(self):
        super()._callbacks()
        # Map query
        if self.open_vocab_query is not None:
            msg = self.toolbox.open_vocab_query(self.open_vocab_query)
            self.open_vocab_query = None
            self.display_llm_response(msg)

    def display_llm_response(
        self,
        msg: str
    ):
        client = self.server.get_clients()[0]
        client.add_notification(
            title="LLM Response",
            body=msg['messages'][-1].content,
            auto_close_seconds=10.0,
        )
            
   