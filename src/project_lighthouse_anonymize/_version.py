"""Version information for project_lighthouse_anonymize."""

try:
    from project_lighthouse_anonymize._version_info import __version__, __version_tuple__
except ImportError:
    __version__ = "unknown"
    __version_tuple__ = (0, 0, 0, "unknown", "unknown")
