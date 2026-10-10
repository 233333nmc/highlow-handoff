"""Namespace for explicitly selected low-level policy plugins.

Concrete plugins are intentionally not imported here.  Import the desired
package (for example ``highlow.inference.low_level.plugins.hyvla``) and
register it at the application composition root.  This keeps discovery from
loading every installed policy integration as the catalog grows.
"""

__all__: list[str] = []
