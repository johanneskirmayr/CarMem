from pydantic import BaseModel


class IncomingPreference(BaseModel):
    vector: list[float]
    text: str
    main_category: str
    subcategory: str
    detail_category: str
    attribute: str
    user_name: str
