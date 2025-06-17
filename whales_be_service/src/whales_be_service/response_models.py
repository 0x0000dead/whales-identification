from pydantic import BaseModel


class Detection(BaseModel):
    image_ind: str
    bbox: list[int]
    class_animal: str
    id_animal: str
    probability: float
    mask: str | None = None # base64 PNG с удалённым фоном
