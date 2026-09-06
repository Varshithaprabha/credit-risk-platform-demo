"""Helpers for paths that differ between local runs and the Docker container."""
import os


def in_docker() -> bool:
    return os.path.exists("/.dockerenv")


def resolve_data_dir(default_local: str, default_docker: str = "/app/data") -> str:
    return default_docker if in_docker() else default_local
