#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

"""Support for masquerading molecule-managed (proxy) views as their owned
equivalents.

Defines the metaclass that lets an owned pybind11 class accept its proxy
counterpart as a virtual subclass, so ``isinstance``/``issubclass`` agree with
the masqueraded type annotations.
"""

from collections import defaultdict
from typing import Dict, Set

__all__ = ["make_virtual_subclass_metaclass"]


def make_virtual_subclass_metaclass(pybind11_cls: type) -> type:
    registry: Dict[type, Set[type]] = defaultdict(set)

    class _VirtualSubclassMeta(type(pybind11_cls)):
        def register(cls, subclass):
            registry[cls].add(subclass)
            return subclass

        def __subclasscheck__(cls, subclass):
            if super().__subclasscheck__(subclass):
                return True

            return issubclass(subclass, tuple(registry.get(cls, ())))

        def __instancecheck__(cls, instance):
            return issubclass(type(instance), cls)

    return _VirtualSubclassMeta
