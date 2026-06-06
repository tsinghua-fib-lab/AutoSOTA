import base64
from abc import ABC, abstractmethod
from enum import Enum
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Union,
    get_args,
)
from urllib.request import urlopen

import PIL.Image
from PIL.ImageFile import ImageFile
from pydantic import BaseModel, Field
from typing_extensions import TypeAlias

from .futures import String, is_string


class ContentPartType(Enum):
    """The types of the content part."""

    TEXT = "text"
    IMAGE = "image_url"
    AUDIO = "input_audio"


class ContentPart(BaseModel, ABC):
    """Represent a part of the message content."""

    type: ContentPartType = Field(..., description="The type of the content part.")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the content part."""
        super().__init__(*args, **kwargs)

    @abstractmethod
    def _get_data(self) -> Union[str, Dict[str, str]]:
        """Return the data of the content part."""

    def get_dict(self) -> Dict[str, Any]:
        """Return a dict representation of the content part."""
        return {"type": self.type.value, self.type.value: self._get_data()}


class TextContent(ContentPart):
    """Represent a text in the message."""

    text: String = Field(..., description="The text content.")

    def __init__(self, text: String) -> None:
        """Initialize the text content."""
        super().__init__(type=ContentPartType.TEXT, text=text)

    def _get_data(self) -> str:
        return str(self.text)

    def __repr__(self) -> str:
        return f"Text(text={self.text})"

    def __str__(self) -> str:
        return f"[Text]{self.text}"


class Image(ContentPart):
    """Represent an image in the message."""

    url: str = Field(
        ..., description="Either a URL of the image or the base64 encoded image data."
    )
    detail: Optional[str] = Field(
        None, description="Specifies the detail level of the image."
    )

    def __init__(self, url: str, detail: Optional[str] = None) -> None:
        """Initialize the image with the URL and detail level.

        See [the guide](https://platform.openai.com/docs/guides/vision/low-or-high-fidelity-image-understanding)
        for more information about the detail level.
        """
        super().__init__(type=ContentPartType.IMAGE, url=url, detail=detail)

    @classmethod
    def from_image(cls, image: ImageFile, detail: Optional[str] = None) -> "Image":
        """Construct an image prompt from a PIL ImageFile."""
        buffered = BytesIO()
        # Save the image to the buffer in PNG format
        image.save(buffered, format="PNG")
        # Get the byte data from the buffer
        img_byte = buffered.getvalue()
        img_base64 = base64.b64encode(img_byte).decode("utf-8")
        return cls(url=f"data:image/png;base64,{img_base64}", detail=detail)

    @classmethod
    def from_file(cls, file: PathLike, detail: Optional[str] = None) -> "Image":
        """Construct an image prompt from an image file."""
        image = PIL.Image.open(Path(file))
        return cls.from_image(image, detail)  # type: ignore

    def _get_data(self) -> Dict[str, Any]:
        d = {"url": self.url}
        if self.detail:
            d["detail"] = self.detail
        return d

    def __repr__(self) -> str:
        return f"Image(url={self.url})"

    def __str__(self) -> str:
        return f"[Image]{self.url}"


class Audio(ContentPart):
    """Represent an audio in the message."""

    data: str = Field(..., description="The base64 encoded audio data.")
    format: Literal["mp3", "wav"] = Field(
        ..., description="The format of the encoded audio data."
    )

    def __init__(self, data: str, format: Literal["mp3", "wav"]) -> None:
        """Initialize the audio with the base64 encoded data and format."""
        super().__init__(type=ContentPartType.AUDIO, data=data, format=format)

    @classmethod
    def from_url(cls, url: str, format: Literal["mp3", "wav"]) -> "Audio":
        """Construct an audio prompt from an audio URL."""
        audio_data = urlopen(url).read()
        base64_encoded = base64.b64encode(audio_data).decode("utf-8")
        return cls(data=base64_encoded, format=format)

    @classmethod
    def from_file(cls, file: PathLike) -> "Audio":
        """Construct an audio prompt from an audio file."""
        ext = Path(file).suffix.lower()[1:]
        if ext not in ("mp3", "wav"):
            raise ValueError(f"Unsupported audio format: {ext}")
        with open(file, "rb") as f:
            audio_data = f.read()
            base64_encoded = base64.b64encode(audio_data).decode("utf-8")
        return cls(data=base64_encoded, format=ext)  # type: ignore

    def _get_data(self) -> Dict[str, Any]:
        return {"data": self.data, "format": self.format}

    def __repr__(self) -> str:
        return f"Audio(data={self.data}, format={self.format})"

    def __str__(self) -> str:
        return f"[Audio]{self.data}"


CONTENT_PART_CLASS_MAP = {
    ContentPartType.TEXT: TextContent,
    ContentPartType.IMAGE: Image,
    ContentPartType.AUDIO: Audio,
}


class ContentList(BaseModel):
    """Represent a list of contents containing text, images and audio."""

    contents: List[ContentPart] = Field(..., description="The content of the message")

    def __init__(
        self, contents: Union[Iterable[ContentPart], Iterator[Dict[str, Any]]]
    ) -> None:
        """Initialize the content list with a list of contents."""
        res: List[ContentPart] = []
        for c in contents:
            if isinstance(c, ContentPart):
                res.append(c)
            elif isinstance(c, dict):
                if "type" not in c:
                    raise ValueError(f"Invalid content: {c}")
                t = ContentPartType(c["type"])
                cls = CONTENT_PART_CLASS_MAP[t]
                kwargs = c[t.value]
                res.append(cls(**kwargs) if isinstance(kwargs, dict) else cls(kwargs))
            else:
                raise ValueError(f"Invalid content: {c}")
        super().__init__(contents=res)

    def __iadd__(
        self, other: Union[ContentPart, List[ContentPart], String, "ContentList"]
    ) -> "ContentList":
        if isinstance(other, ContentList):
            self.extend(other.contents)
        elif isinstance(other, ContentPart):
            self.append(other)
        elif isinstance(other, list):
            self.extend(other)
        elif isinstance(other, get_args(String)):
            self.append(TextContent(other))
        else:
            raise ValueError(f"Invalid content: {other}")
        return self

    def append(self, content: ContentPart) -> None:
        """Append a content to the list.

        If the last content is a string, it will be concatenated with the new content.
        """
        if is_string(content) and len(self.contents) and is_string(self.contents[-1]):
            self.contents[-1] += content  # type: ignore
        else:
            self.contents.append(content)

    def extend(self, contents: list[ContentPart]) -> None:
        """Extend the list with multiple contents."""
        for content in contents:
            self.append(content)

    def get_contents(self) -> List[Dict[str, Any]]:
        """Return the contents as a list of dictionaries."""
        return [c.get_dict() for c in self.contents]

    def __str__(self) -> str:
        return "\n".join([str(c) for c in self.contents])
