# Guía de contribución

Este documento resume el flujo de trabajo del equipo de **ArequipaHouse ML**
(capítulo 5 del informe del proyecto).

## Ramas

`main` siempre contiene código que funciona. **No se hacen cambios directamente
en `main`**: todo cambio entra mediante un Pull Request.

| Rama | Uso | Ejemplo |
|---|---|---|
| `main` | Código estable y desplegable | `main` |
| `feature/…` | Funcionalidad nueva | `feature/comparador-viviendas` |
| `fix/…` | Corrección de un error | `fix/horario-visitas` |
| `docs/…` | Cambios solo de documentación | `docs/readme-actualizado` |
| `chore/…` | Mantenimiento y configuración | `chore/plantillas-github` |

## Flujo de trabajo

1. Crear o elegir un *issue* que describa la tarea.
2. Actualizar `main` y crear la rama de trabajo:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/nombre-corto
   ```
3. Programar y probar en local:
   ```bash
   python -m pytest
   ```
4. Hacer commit y subir la rama:
   ```bash
   git add -A
   git commit -m "feat: descripción corta del cambio"
   git push -u origin feature/nombre-corto
   ```
5. Abrir un Pull Request hacia `main`, indicando el issue relacionado
   (por ejemplo, `Closes #1`).
6. Solicitar la revisión de al menos otro integrante del equipo.
7. Cuando la revisión está aprobada y las pruebas pasan, fusionar con
   **Squash and merge** y eliminar la rama.

## Mensajes de commit

Se escriben en español, en modo imperativo y con un prefijo que indica el tipo
de cambio:

- `feat:` funcionalidad nueva
- `fix:` corrección de un error
- `docs:` documentación
- `test:` pruebas
- `refactor:` reestructuración sin cambiar el comportamiento
- `chore:` tareas de mantenimiento

Un commit contiene un solo cambio lógico y deja las pruebas en verde.

## Conflictos

Si hay conflictos con `main`, se resuelven en la rama de trabajo antes de pedir
la fusión:

```bash
git fetch origin
git merge origin/main
```

## Pruebas

La integración continua (`.github/workflows/ci.yml`) ejecuta las pruebas con
SQLite y con PostgreSQL 16 en cada *push* y Pull Request.
