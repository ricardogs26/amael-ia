from pydantic import BaseModel

class OrganizeDayResponse(BaseModel):
    summary: str
    tasks_created: int
