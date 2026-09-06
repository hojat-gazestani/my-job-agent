from pydantic import BaseModel, HttpUrl


class JobMatch(BaseModel):
    title: str
    company: str
    location: str
    fit_score: int
    visa_status: str
    url: HttpUrl
    reason: str


class JobReport(BaseModel):
    jobs: list[JobMatch]
