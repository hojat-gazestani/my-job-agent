from pydantic import BaseModel, Field


class GeneratedQueries(BaseModel):
    """
    Pydantic Schema for Planner
    """

    queries: list[str] = Field(
        description="List of targeted queries derived from candidate profile."
    )
