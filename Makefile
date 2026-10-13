VENV   := .venv
PORT   ?= 8000
WEB    := apps/web

ifeq ($(OS),Windows_NT)
PYTHON := $(VENV)/Scripts/python.exe
PY_BOOTSTRAP := python
else
PYTHON := $(VENV)/bin/python
PY_BOOTSTRAP := python3
endif

UVICORN := $(PYTHON) -m uvicorn main:app --app-dir apps/api

.PHONY: help install build start api web dev

help:
	@echo "make install   Tạo .venv, cài requirements.txt và thư viện của web (npm)"
	@echo "make start     Build giao diện rồi chạy app tại http://localhost:$(PORT)"
	@echo "make dev       Chạy API (cổng 8000) và Vite (cổng 5173) song song, tự tải lại khi sửa code"
	@echo "make api       Chỉ chạy API FastAPI, tài liệu API tại /docs"
	@echo "make web       Chỉ chạy Vite dev server"
	@echo "make build     Chỉ build giao diện vào $(WEB)/dist"

$(PYTHON):
	$(PY_BOOTSTRAP) -m venv $(VENV)

install: $(PYTHON)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	npm --prefix $(WEB) ci

build:
	npm --prefix $(WEB) run build

# App tự báo lỗi nếu thiếu checkpoint/dataset.
start: build
	$(UVICORN) --port $(PORT)

api: $(PYTHON)
	$(UVICORN) --port 8000 --reload --reload-dir apps/api

web:
	npm --prefix $(WEB) run dev

dev:
	$(MAKE) -j2 api web
