from fastapi import APIRouter

router = APIRouter(
    tags=["root"],
)


@router.get("/")
def root():
    return {
        "describe": "API for recommendations system",
        "version": "0.0.1b",
        "docs": "/docs",
    }
