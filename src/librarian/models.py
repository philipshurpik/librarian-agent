import html
import re

from pydantic import BaseModel, field_validator

_BREAK_TAG_RE = re.compile(r'</p\s*>|<br\s*/?>', re.IGNORECASE)
_TAG_RE = re.compile(r'<[^>]+>')
_YEAR_RE = re.compile(r'\d{4}')


def clean_text(text: str) -> str:
    """Replace break (br, p) tags with new line + remove other tags + unicode convert + clean spaces and new lines"""
    text = _TAG_RE.sub('', _BREAK_TAG_RE.sub('\n\n', text))
    text = html.unescape(text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\s*\n\s*\n\s*', '\n\n', text)
    return text.strip()


class BookAttributes(BaseModel):
    author: str
    topic: str
    year: int | None = None
    level: str | None = None

    @field_validator('author', 'topic', 'level')
    @classmethod
    def collapse_whitespace(cls, v: str | None) -> str | None:
        return ' '.join(v.split()) if v else v

    @field_validator('year', mode='before')
    @classmethod
    def coerce_year(cls, v: int | str | None) -> int | None:
        if isinstance(v, str):  # e.g. 'c2013', '2016'
            match = _YEAR_RE.search(v)
            return int(match.group()) if match else None
        return v


class Book(BaseModel):
    id: str
    title: str
    attributes: BookAttributes
    description: str | None = None
    available_units: int = 0

    @field_validator('title')
    @classmethod
    def collapse_whitespace(cls, v: str) -> str:
        return ' '.join(v.split())

    @field_validator('description')
    @classmethod
    def clean_description(cls, v: str | None) -> str | None:
        return clean_text(v) or None if v else None

    @field_validator('available_units')
    @classmethod
    def clamp_units(cls, v: int) -> int:
        return max(0, v)

    @property
    def dedupe_key(self) -> tuple[str, str]:
        return self.title.lower(), self.attributes.author.lower()
