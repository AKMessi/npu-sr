class SRException(Exception):
    """An actionable error suitable for displaying without a traceback."""


class NPUUnavailable(SRException):
    """The strict NPU backend could not be established."""
