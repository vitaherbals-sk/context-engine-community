FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e ".[all]" 2>/dev/null || pip install --no-cache-dir -e .

# Create data directory for SQLite volume
RUN mkdir -p /data

# Default environment
ENV CTX_DB=/data/context-engine.db
ENV CTX_HOST=0.0.0.0
ENV PYTHONPATH=/app/src

# Force UTF-8 everywhere (stdout, filesystem, locale) so non-ASCII content —
# Slovak diacritics in notes, names, log lines — never depends on the image's
# default locale, which python:slim leaves as POSIX.
ENV PYTHONUTF8=1
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Expose port
EXPOSE 8000

# Run combined OAuth + MCP server
CMD ["python", "-m", "context_engine.combined_server"]
