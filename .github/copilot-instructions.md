## Objetivo rápido

Este archivo dirige a un agente de codificación (Copilot / agente AI) a ser inmediatamente útil en este repositorio. El árbol actual del proyecto está vacío; por tanto las instrucciones se enfocan en pasos de descubrimiento, convenciones esperadas y reglas seguras para crear cambios mínimos y verificables.

## Estado del repositorio

- Actualmente no hay archivos en el repositorio raíz. Antes de cambiar nada, sigue la sección "Primeros pasos de descubrimiento".

## Primeros pasos de descubrimiento (obligatorio)

1. Ejecuta un listado y búsqueda rápida para detectar gestores de paquetes, scripts de build y ficheros CI:

```bash
# listar archivos tracked
git ls-files --exclude-standard

# buscar archivos comunes
rg "package.json|pyproject.toml|setup.py|requirements.txt|Pipfile|go.mod|Cargo.toml|Makefile|Dockerfile|.github/workflows" || true
``` 

2. Si encuentras un `package.json`, abre `scripts` y anota comandos relevantes (`start`, `build`, `test`). Para Python busca `pytest` o `tox` en `pyproject.toml` o `requirements.txt`.

3. Busca convenciones en carpetas: `src/`, `app/`, `cmd/`, `internal/`, `pkg/`, `api/`, `services/`. Si existe `README.md` lo priorizas para entender el propósito.

## Cómo tomar decisiones de diseño pequeñas

- Si el repo está vacío o faltan tests, crea un cambio minimalista: agregar README inicial, `LICENSE` (si corresponde), y un archivo de configuración de CI básico sólo si el mantenedor lo solicita.
- Prefiere cambios que no alteren infra/producción (no tocar `infra/` o `deploy/` sin permiso explícito).

## Convenciones de commit / PR

- Mensajes de commit: prefijo corto y claro: `fix:`, `feat:`, `chore:`, `docs:` seguido de descripción de 1 línea.
- Cada PR debe incluir: descripción corta, lista de cambios, cómo probar localmente, y pruebas automáticas añadidas cuando sea necesario.

## Comandos útiles (ejemplos de descubrimiento)

```bash
# inspección rápida del repo
ls -la
rg "TODO|FIXME|HACK" || true

# ejecutar tests (si detectas pytest / npm / go)
# npm: npm install && npm test
# python: python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && pytest -q
``` 

## Reglas y límites del agente

1. No enviar cambios que incluyan credenciales o archivos `.env` con secretos. Si encuentras secretos, notifica y crea una tarea para rotación.
2. No modificar configuraciones de producción o despliegue sin la aprobación del mantenedor humano.
3. Evita añadir dependencias de terceros nuevas sin justificar el motivo y bajas implicaciones de seguridad.

## Qué documentar en el PR o commit si haces cambios

- Por qué fue necesario el cambio.
- Cómo reproducir localmente (comandos concretos).
- Archivos clave tocados y su propósito.

## Preguntas que el agente debe plantear al humano (cuando aplique)

- ¿Cuál es el lenguaje/plataforma objetivo principal (Node/Python/Go/Rust)?
- ¿Hay tests obligatorios a mantener y ejecutar en CI? ¿Algún runner específico?
- ¿Puedo añadir un `README.md` inicial o un pipeline CI de ejemplo si el repo no tiene nada?

## Contactos y referencias

- Si existe un `CONTRIBUTING.md` o `MAINTAINERS` en el repo, sigue esas instrucciones antes de proponer cambios.

---
Si alguna sección está incompleta o hay información privada que no puedo leer, pídeme que busque archivos/paths específicos o comparte el README para adaptar estas instrucciones. ¿Quieres que añada un `README.md` inicial y un pipeline CI básico como PR de ejemplo?
