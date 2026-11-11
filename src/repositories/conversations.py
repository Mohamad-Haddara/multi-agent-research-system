"""
Implementing the conversation repository using abstract repository interface.
"""

from src.database.entities import Conversation
from repositories.interfaces import Repository
from src.schemas.database import ConversationCreate, ConversationUpdate

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession




# Inherit the abstract Repository interface and implement each of its methods
class ConversationRepository(Repository):

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list(self, skip: int, take:int) -> list[Conversation]:

        async with self.session.begin():
            result = await self.session.execute(
                select(Conversation).offset(skip).limit(take)
            )

        return [r for r in result.scalar().all()]

    

    async def get(self, conversation_id: int) -> Conversation | None:

        async with self.session.begin():
            result = await self.session.execute(
                select(Conversation).where(Conversation.id == conversation_id)
            )

        return result.scalar().first()


    
    async def create(self, conversation: ConversationCreate) -> Conversation:
        new_conversation = Conversation(**conversation.model_dump())
        async with self.session.begin():
            self.session.add(new_conversation)
            await self.session.commit()
            await self.session.refresh(new_conversation)


    

    async def update(
        self, conversation_id: int, updated_conversation: ConversationUpdate
    ) -> Conversation | None:

        conversation = await self.get(conversation_id)

        if not conversation:
            return None

        
        for key, value in updated_conversation.model_dump():
            setattr(conversation, key, value)

        async with self.session.begin():

            await self.session.commit()
            await self.session.refresh(conversation)

        return conversation


    

    async def delete(self, conversation_id: int) -> None:
        conversation = self.get(conversation_id)

        if not conversation:
            return

        async with self.session.begin():
            await self.session.delete(conversation)
            await self.session.commit()