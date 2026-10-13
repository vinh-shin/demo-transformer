VENV   := .venv
PORT   ?= 8501

ifeq ($(OS),Windows_NT)
PYTHON := $(VENV)/Scripts/python.exe
PY_BOOTSTRAP := python
else
PYTHON := $(VENV)/bin/python
PY_BOOTSTRAP := python3
endif

.PHONY: start install help

help:
	@echo "make install   Tạo .venv và cài requirements.txt"
	@echo "make start     Chạy app demo Streamlit (PORT=$(PORT))"

$(PYTHON):
	$(PY_BOOTSTRAP) -m venv $(VENV)

install: $(PYTHON)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# App tự báo lỗi nếu thiếu checkpoint/dataset.
start: $(PYTHON)
	$(PYTHON) -m streamlit run app/app.py --server.port $(PORT)
