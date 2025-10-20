#!/bin/bash
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --limit-concurrency 10 --workers 1
