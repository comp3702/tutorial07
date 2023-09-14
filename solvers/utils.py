from typing import Dict

from envs.GridWorldWithKeys import EXIT_STATE, ACTION_NAMES


def dict_argmax(dictionary: Dict):
    max_value = max(dictionary.values()) # TODO handle multiple keys with the same max value
    for key, value in dictionary.items():
        if value == max_value:
            return key

class VisualizerMixin():
    def get_values_and_policy(self):
        data = []
        for state, value in self.state_values.items():
            # if state == EXIT_STATE:
            #     continue
            desc = ''
            if state.position() in state.key_state:
                desc = '\nKey'
            if state.position() in self.env.hazards:
                desc = '\nHaz'
            if not state.key_state and state.position() in self.env.goal:
                desc = '\nGoal'

            data.append( (state.x, state.y, ACTION_NAMES[self.policy[state]], value, desc, state.key_state) )

        return data

    def print_values_and_policy(self):
        for state, value in self.state_values.items():
            print(state, ACTION_NAMES[self.policy[state]], value)

    def print_values(self):
        for state, value in self.state_values.items():
            print(state, value)

    def print_policy(self):
        for state, policy in self.policy.items():
            print(state, policy)