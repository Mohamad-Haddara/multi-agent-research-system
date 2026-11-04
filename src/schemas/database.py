"""
Pydantic API schemas for conversation endpoints
"""

from datetime import datetime
from pydantic import BaseModel, ConfigDict


# Set up the Pydantic model to read and validate attributes of other models like SQLAlchemy, which is often used in Pydantic when working with database models
class ConversationBase(BaseModel):
   
    model_config = ConfigDict(from_attributes=True)

    title: str
    model_type: str




# Create a separate Pydantic models based on the base model for different use cases such as conversation record creation and update or data retrieval
class ConversationCreate(ConversationBase):
    pass 



class ConversationUpdate(ConversationBase):
    pass


class ConversationOut(ConversationBase):
    id: int
    created_at: datetime
    updated_at: datetime

