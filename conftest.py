"""Pytest conftest: configuración compartida para todos los tests.

No mockea los modelos automáticamente — cada archivo de test
gestiona sus propios mocks según necesite.
"""

import sys
from pathlib import Path

# Asegurar que el directorio raíz del proyecto está en sys.path
sys.path.insert(0, str(Path(__file__).parent))
