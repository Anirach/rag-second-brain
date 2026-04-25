"""RAG Second Brain - Multi-source retrieval with learned gating.

Keep package import light.

The root test suite imports modules like ``src.cooccurrence`` and ``src.gating``
individually. Eagerly importing heavyweight subpackages here pulls in optional
KG / ML dependencies at import time and breaks unrelated tests in lean
environments.
"""

__version__ = "0.16.0"
__author__ = "Anirach Mingkhwan"

__all__ = ["retrieval", "fusion", "evaluation", "utils"]
