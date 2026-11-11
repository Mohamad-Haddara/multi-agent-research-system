from abc import ABC, abstractmethod
from re import A
from typing import Any



# Define the abstract Repository interface with several CRUD related abstract method that subclasses must implement
# If an abstract method is not implemented in a concrete subclass; NotImplementedError will be raised.
class Repository(ABC):

    @abstractmethod
    async def list(self) -> list[Any]:
        pass


    @abstractmethod
    async def get(self, uid: int) -> Any:
        pass


    @abstractmethod
    async def create(self, record: Any) -> Any:
        pass


    @abstractmethod
    async def update(self, uid: int, record: Any) -> Any:
        pass


    @abstractmethod
    async def delete(seld, uid: int) -> Any:
        pass
