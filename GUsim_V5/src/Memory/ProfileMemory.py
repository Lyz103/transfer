import os
import sys
from abc import ABC, abstractmethod
from Memory.BaseMemory import ExplicitMemory
from Memory.Recall import ProfileMemoryRecall
from Memory.Store  import ProfileMemoryStore
from Memory.utils.Storage import LinearStorage

class ProfileMemory(ExplicitMemory):
    """
    Chat Memory to save User's chat history.
    """
    def __init__(self, config) -> None:
        super().__init__(config)

        self.storage = LinearStorage(self.config.args.storage)
        self.store_op = ProfileMemoryStore(self.config.args.store, storage = self.storage)
        self.recall_op = ProfileMemoryRecall(self.config.args.recall, storage = self.storage)

        # self.auto_display = eval(self.config.args.display.method)(self.config.args.display, register_dict = {
        #     'Memory Storage': self.storage
        # })
    
    def reset(self) -> None:
        self.__reset_objects__([self.storage, self.store_op, self.recall_op])

    def store(self, observation) -> None:
        self.store_op(observation)
    
    def recall(self, query) -> object:
        return self.recall_op(query)
    
    def display(self) -> None:
        # self.auto_display(self.storage.counter)
        print(self.storage.display())

    def manage(self, operation, **kwargs) -> None:
        pass
    
    def optimize(self, **kwargs) -> None:
        pass


if __name__ == "__main__":
    # sys.path[0] = os.path.join(os.path.dirname(__file__), '..', '..')
    from modules.Memory.default_config.DefaultMemoryConfig import DEFAULT_FUMEMORY
    from modules.Memory.config.Config import MemoryConfig
    def ActionMemory1():
        memory_config = MemoryConfig(DEFAULT_FUMEMORY)
        memory = ChatMemory(memory_config)
        # print("Memory Config:", memory_config.args.recall.utilization.method)
        memory.store('User: [Eat]')
        memory.store('Alice holds a master\'s degree in English Literature.')
        # memory.reset()
        # memory.display()
        memory.store('Alice loves reading and jogging.')
        memory.store('Alice has a pet cat named Whiskers.')
        memory.store('Last year, Alice traveled to New York to attend a literary conference.')
        memory.store('Bob is Alice\'s best friend, who is an excellent engineer.')
        # memory.display()
        print(memory.recall('What'))

    ActionMemory1()