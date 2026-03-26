import pydantic

from appl.caching.utils import dict_to_pydantic, pydantic_to_dict


def test_pydantic_conversion():
    class TestModel(pydantic.BaseModel):
        name: str
        age: int

    data = {"name": "John", "age": 30}
    model = dict_to_pydantic(data, TestModel)
    assert model.name == data["name"]
    assert model.age == data["age"]

    new_data = pydantic_to_dict(model)
    assert new_data == data
