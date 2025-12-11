"""Chat interface."""


def __getattr__(name: str):
    """Lazy import to avoid RuntimeWarning when running as __main__."""
    if name == "ChatInterface":
        from hologram.chat.interface import ChatInterface
        return ChatInterface
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ChatInterface"]
