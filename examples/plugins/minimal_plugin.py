"""Minimal external AGU plugin registration example.

Package this module in another distribution and declare:

    [project.entry-points."agu.plugins"]
    example = "your_package.plugin:register"
"""

from app.analysis.pipeline import CallableStage, register_analysis_stage
from app.plugins import PluginRegistry, PluginSpec


def register(registry: PluginRegistry) -> None:
    def annotate(context) -> None:
        context.metadata["example_plugin"] = "active"

    register_analysis_stage(
        CallableStage("example.annotate", annotate),
        position="after_dispatch",
        replace=True,
    )
    registry.register(
        PluginSpec(
            name="example-review-export",
            kind="integration",
            capabilities=("review-export",),
            version="0.1.0",
            source="example-plugin",
            description="Demonstrates metadata-only AGU plugin registration.",
        )
    )
