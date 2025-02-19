class ErtError(Exception):
    """Base class for exceptions in this module."""

    pass


class IllegalWorkspaceOperation(ErtError):
    def __init__(self, message: str) -> None:
        self.message = "{}".format(message)


class IllegalWorkspaceState(ErtError):
    def __init__(self, message: str) -> None:
        self.message = "{}".format(message)


class NonExistantExperiment(IllegalWorkspaceOperation):
    def __init__(self, message: str) -> None:
        self.message = "{}".format(message)


class ConfigValidationError(ErtError):
    def __init__(self, message: str, source: str = None) -> None:
        self.message = "{}".format(message)
        self.source = source


class StorageError(ErtError):
    def __init__(self, message: str) -> None:
        self.message = "{}".format(message)
