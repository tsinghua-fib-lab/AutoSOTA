from pydantic import BaseModel

from appl import gen, ppl


# Define your desired output structure
class UserInfo(BaseModel):
    name: str
    age: int


@ppl
def get_user_info() -> UserInfo:
    # Extract structured data from natural language
    "John Doe is 30 years old."
    return gen(response_format=UserInfo).results


@ppl
def get_user_info_instructor() -> UserInfo:
    # Extract structured data from natural language
    "John Doe is 30 years old."
    return gen(response_model=UserInfo).results


print("Using response_format:")
user_info = get_user_info()

print(user_info.name)
# > John Doe
print(user_info.age)
# > 30

try:
    import instructor

    print("Using Instructor's response_model:")
    user_info = get_user_info_instructor()

    print(user_info.name)
    # > John Doe
    print(user_info.age)
    # > 30
except (ImportError, ModuleNotFoundError):
    print("Instructor is not installed, skipping instructor example.")
