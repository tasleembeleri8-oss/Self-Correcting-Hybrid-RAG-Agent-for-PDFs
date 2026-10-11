from pydantic import BaseModel, Field


class Grade(BaseModel):
    """Structured output of the relevance grader (forces the LLM to answer true/false)."""
    relevant: bool = Field(description="True if the passages contain enough information to answer the question")


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int


class AskRequest(BaseModel):
    # 32 hex characters = uuid4().hex. Rejecting anything else blocks malformed or probing ids early.
    document_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    question: str = Field(min_length=3, max_length=500)


class AskResponse(BaseModel):
    answer: str
    retries: int
    passages: list[str] = []
