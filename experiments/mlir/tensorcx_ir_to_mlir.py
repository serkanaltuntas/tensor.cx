"""Compatibility entry point for the research scripts; emission has one owner."""
from tensorcx._compiler.emitter import MlirEmitError, emit_mlir, format_f32_constant

__all__ = ["MlirEmitError", "emit_mlir", "format_f32_constant"]
