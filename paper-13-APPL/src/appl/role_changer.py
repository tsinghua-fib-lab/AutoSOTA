from typing import Any, Optional

from typing_extensions import override

from .core.context import PromptContext
from .core.modifiers import PrinterModifier, PrinterPush
from .core.types import MessageRole, MessageRoleType


class RoleChanger(PrinterModifier):
    """The contextual role changer of the prompts."""

    _new_role: Optional[MessageRole] = None

    def __init__(
        self, role: Optional[MessageRole] = None, _ctx: Optional[PromptContext] = None
    ):
        """Initialize the RoleChanger object.

        Args:
            role: The new role of the prompts. Defaults to None.
            _ctx: The prompt context filled automatically by the APPL function.
        """
        super().__init__(_ctx)
        if role is not None:
            self._new_role = role

    @override
    @property
    def push_args(self) -> PrinterPush:
        return PrinterPush(new_role=self._new_role)


class SystemRole(RoleChanger):
    """Change the role of the prompts to system."""

    def __init__(self, name: Optional[str] = None, **kwargs: Any):
        """Initialize the SystemRole object.

        Args:
            name: The name of the system role. Defaults to None.
            **kwargs: The keyword arguments to pass to the RoleChanger constructor.
        """
        role = MessageRole(MessageRoleType.SYSTEM, name=name)
        super().__init__(role=role, **kwargs)


class UserRole(RoleChanger):
    """Change the role of the prompts to user."""

    def __init__(self, name: Optional[str] = None, **kwargs: Any):
        """Initialize the UserRole object.

        Args:
            name: The name of the user role. Defaults to None.
            **kwargs: The keyword arguments to pass to the RoleChanger constructor.
        """
        role = MessageRole(MessageRoleType.USER, name=name)
        super().__init__(role=role, **kwargs)


class AIRole(RoleChanger):
    """Change the role of the prompts to assistant."""

    def __init__(self, name: Optional[str] = None, **kwargs: Any):
        """Initialize the AIRole object.

        Args:
            name: The name of the assistant role. Defaults to None.
            **kwargs: The keyword arguments to pass to the Role
        """
        role = MessageRole(MessageRoleType.ASSISTANT, name=name)
        super().__init__(role=role, **kwargs)


class ToolRole(RoleChanger):
    """Change the role of the prompts to tool."""

    def __init__(self, name: Optional[str] = None, **kwargs: Any):
        """Initialize the ToolRole object.

        Args:
            name: The name of the tool role. Defaults to None.
            **kwargs: The keyword arguments to pass to the RoleChanger constructor.
        """
        role = MessageRole(MessageRoleType.TOOL, name=name)
        super().__init__(role=role, **kwargs)
