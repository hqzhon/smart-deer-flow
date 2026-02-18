from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    output: Any = Field(default=None)
    error: str | None = Field(default=None)
    base64_image: str | None = Field(default=None)
    system: str | None = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __bool__(self):
        return any(getattr(self, field) for field in self.model_fields)

    def __add__(self, other: "ToolResult"):
        def combine_fields(
            field: str | None, other_field: str | None, concatenate: bool = True
        ) -> str | None:
            if field and other_field:
                if concatenate:
                    return field + other_field
                raise ValueError("Cannot combine tool results")
            return field or other_field

        return ToolResult(
            output=combine_fields(self.output, other.output),
            error=combine_fields(self.error, other.error),
            base64_image=combine_fields(self.base64_image, other.base64_image, False),
            system=combine_fields(self.system, other.system),
        )

    def __str__(self):
        return f"Error: {self.error}" if self.error else str(self.output)

    def replace(self, **kwargs) -> "ToolResult":
        return type(self)(**{**self.model_dump(), **kwargs})


class CLIResult(ToolResult):
    pass


class ToolFailure(ToolResult):
    pass
