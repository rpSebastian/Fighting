from malib.agent import MAPlayer

import numpy as np


class SCPlayer(MAPlayer):
    def make_mask(self, obs):
        mask_to_use = {}
        mask_raw = obs["avail_actions"]
        # mask_data = {}
        # for m_k, m_agents in self.model2agent.items():
        #     mask_to_use[m_k] = []
        #     for agent_id in m_agents:
        #         mask_np = np.array(mask_raw[agent_id])
        #         mask_to_use[m_k].append(mask_np)
        #         mask_data[agent_id] = mask_np

        # return mask_to_use, mask_data
        return mask_raw
