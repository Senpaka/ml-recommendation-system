from pydantic import BaseModel, Field, field_validator


class RecommendationItem(BaseModel):
    movie_id: int
    score: float


class RecommendationResponse(BaseModel):
    recommendations: list[RecommendationItem]


class UserRequest(BaseModel):
    user_id: int = Field(ge=0, description="User ID")
    gender: str | None = Field(max_length=1, description="User gender: M/F")
    age_group: int | None = Field(description="Age group for user")

    @field_validator("gender")
    @classmethod
    def gender_is_m_f_or_none(cls, v: str) -> str:
        if v is None or v in ["M", "F"]:
            return v.strip()

        raise ValueError("Gender must be M, F or None")
