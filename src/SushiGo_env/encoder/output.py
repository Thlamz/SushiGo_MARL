from typing import TypedDict


class ActorOutputKeys(TypedDict):
    flat: dict[str, tuple[str, ...]]
    sequential: dict[str, tuple[str, ...]]