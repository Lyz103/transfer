import enum
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict


@dataclass(frozen=True)
class Action:
    """An action that the agent can take."""

    name: str
    description: str
    content: Optional[str] = None
    flexible_reward: bool = False

    def __str__(self):
        return f"{self.name}: {self.description}"


class Reward:
    """The reward of the agent"""

    def __init__(self, reward: dict[str, int] = None, inner_voice: dict[str, str] = None, flexible: bool = True):
        self.flexible = flexible
        self.reward = reward.copy() if reward is not None else {}
        self.inner_voice = inner_voice.copy() if inner_voice is not None else {}

    def update_reward(self, reward: dict[str, int]):
        self.reward.update(reward)

    def update_inner_voice(self, inner_voice: dict[str, str]):
        self.inner_voice.update(inner_voice)

    def __str__(self):
        return f"{self.reward}"

    def __repr__(self):
        return self.__str__()

    def get_reward(self):
        # return the average reward
        if len(self.reward) == 0:
            return 0
        return sum(self.reward.values()) / len(self.reward)


class Tool:
    """the base class for the tools"""

    def __init__(self, name: str, description: str, all_user_info: bool = False):
        self.name = name
        self.description = description
        self.all_user_info: bool = False


class Agent:
    """The base class for the agent"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def interact(self, str_input: Union[str, None]) -> str:
        raise NotImplementedError

    def clear_interaction_history(self):
        raise NotImplementedError

    def __call__(self, str_input: Union[str, None] = None) -> str:
        return self.interact(str_input)


class Counter:
    """The counter for evaluation."""

    def __init__(self, name: str):
        self.name = name
        self.counter = {}

    def update(self, elements: list):
        for element in elements:
            if element not in self.counter:
                self.counter[element] = 1
            else:
                self.counter[element] += 1

    def update_dict(self, elements: dict):
        for key, value in elements.items():
            if key not in self.counter:
                self.counter[key] = value
            else:
                self.counter[key] += value

    def get_counter(self):
        return self.counter

    def __str__(self):
        return f"Class Counter for {self.name}: {self.counter}"


class SimulatedUserAct:
    """The output of the simulated user."""

    def __init__(self, action: List[Action], user_response: str, reward=None):
        self.action = action
        self.user_response = user_response
        self.reward = reward

    def __str__(self):
        return f"Action names: {[x.name for x in self.action]}, \n" \
               f"User response: {self.user_response}, \n" \
               f"Reward: {self.reward}"


class SingleProperty:
    def __init__(self, name: str, options: List, sample_policy: Dict, flexible: bool = True,
                 conflict: List[str] = None,
                 refiner=None
                 ):
        """

        :param name: the name of the property
        :param options: the options of the property
        :param sample_policy: the sample policy of the property{
                    "policy": choose from ["fixed", "random", "possibility"]
                    "possibility": a dict of the possibility of each option, the missing options will be set to (1 - sum(possibility.values())) / len(options)
                    "random_num": if policy is "random", the number of options to choose
                }
        :param flexible: whether the property is flexible
        :param conflict: the conflict properties, the sampling process will avoid the conflict properties
        :param refiner: the refiner for the property, if the refiner is not None, the property will be refined
        """
        self.name = name
        self.options = options
        self.flexible = flexible
        self.emphasis = False
        self.conflict = conflict
        self.sample_policy = sample_policy
        self.refiner = refiner

        self.values = self.sample(sample_policy, conflict)

    def __str__(self):
        if self.emphasis:
            return f"{self.name}(EMPHASIS): {self.values}"
        else:
            return f"{self.name}: {self.values}"

    def __repr__(self):
        return self.__str__()

    def set_refiner(self, refiner):
        if refiner.property_name != self.name:
            raise ValueError("The property name does not match the refiner")
        self.refiner = refiner

    def set_emphasis(self, emphasis: bool):
        self.emphasis = emphasis

    def sample(self, policy: Dict, conflict_value: List[str] = None):

        if conflict_value is None:
            conflict_value = []
        if policy is None:
            raise ValueError("policy not found")

        if policy["policy"] == "random_one":
            if "possibility" in policy:
                # sample according to the possibility
                # 获取概率 {"无": 0.8} "饮食限制": ["无", "素食主义者", "生酮饮食", "高蛋白"],
                posibility = policy["possibility"]
                values = self.options.copy()
                values = [x for x in values if x not in conflict_value]
                # the values that are not in the possibility, sample uniformly
                remain_posibility = 1 - sum(posibility.values())
                p = []
                for value in values:
                    if value in posibility:
                        p.append(posibility[value])
                    else:
                        p.append(remain_posibility / len(values))
                sampled_value = random.choices(values, p)[0]

            else:
                # sample uniformly
                values = self.options.copy()
                values = [x for x in values if x not in conflict_value]
                sampled_value = random.choice(values)

            return [sampled_value]

        elif policy["policy"] == "random":
            if "random_num" in policy:
                random_num = policy["random_num"]
                if type(random_num) is not list:
                    raise ValueError("random_num should be a list")
                if len(random_num) != 2:
                    raise ValueError("random_num should have 2 elements, [min, max]")
                values = self.options.copy()
                values = [x for x in values if x not in conflict_value]
                sampled_value = random.sample(values, random.randint(random_num[0], random_num[1]))
                return sampled_value
            else:
                raise ValueError("Using \"random\" policy, but \"random_num\" not found")

        elif policy["policy"] == "all":
            values = self.options.copy()
            values = [x for x in values if x not in conflict_value]
            return values
        else:
            raise ValueError("policy not found")

    def change_values(self, value: list):
        for v in value:
            if v not in self.options:
                raise ValueError(f"{v} not in the options")
        self.values = value

    def get_values(self):
        """
        return the values of the property
        :return: list
        """
        return self.values


class UnifiedProperty(SingleProperty):
    def __init__(self, name: str, options: List[SingleProperty], sample_policy: Dict, flexible: bool = True,
                 conflict: List[str] = None):
        """
        :param name: the name of the property
        :param options: the options of the property
        :param sample_policy: the sample policy of the property{
                    "policy": choose from ["fixed", "random", "possibility"]
                    "possibility": a dict of the possibility of each option, the missing options will be set to (1 - sum(possibility.values())) / len(options)
                    "random_num": if policy is "random", the number of options to choose
                }
        :param flexible: whether the property is flexible
        :param conflict: the conflict properties, the sampling process will avoid the conflict properties
        """
        sample_options = []
        for option in options:
            sample_options.append(option.name)
        super().__init__(name, sample_options, sample_policy, flexible, conflict)
        self.properties = []
        for value in self.values:
            for option in options:
                if value in option.name:
                    self.properties.append(option)
                    break

    def __str__(self):
        if self.emphasis:
            return f"{self.name}(EMPHASIS): {[x for x in self.properties]}"
        else:
            return f"{self.name}: {[x for x in self.properties]}"

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, item):
        try:
            for option in self.properties:
                if option.name == item:
                    return option
        except IndexError:
            return None

    def __iter__(self):
        return iter(self.properties)

    def get_values(self):
        """
        return the values of the property, for unified property the return value is a dict
        for single property the return value is a list
        use recursive method
        :return: dict
        """
        return {x.name: x.get_values() for x in self.properties}
