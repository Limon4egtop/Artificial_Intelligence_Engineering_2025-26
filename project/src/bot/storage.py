from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ModerationStatus(str, Enum):
    NEW = "новое"
    CHECKED = "проверено"


@dataclass
class ModerationCase:
    case_id: str
    source_chat_id: int
    chat_title: str | None
    source_message_id: int
    user_id: int
    username: str | None
    user_link: str
    text: str
    label: str
    confidence: float
    action: str
    created_at: datetime
    status: ModerationStatus = ModerationStatus.NEW
    checked_by: int | None = None
    checked_by_username: str | None = None
    moderator_message_id: int | None = None


class InMemoryModerationStorage:
    def __init__(self) -> None:
        self._cases: dict[str, ModerationCase] = {}

    def add(self, case: ModerationCase) -> None:
        self._cases[case.case_id] = case

    def get(self, case_id: str) -> ModerationCase | None:
        return self._cases.get(case_id)

    def mark_checked(
            self,
            case_id: str,
            moderator_id: int,
            moderator_username: str | None = None,
    ) -> None:
        case = self._cases[case_id]
        case.status = ModerationStatus.CHECKED
        case.checked_by = moderator_id
        case.checked_by_username = moderator_username