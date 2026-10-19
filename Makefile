VENV   := .venv

ifeq ($(OS),Windows_NT)
PYTHON := $(VENV)/Scripts/python.exe
PY_BOOTSTRAP := python
else
PYTHON := $(VENV)/bin/python
PY_BOOTSTRAP := python3
endif

.PHONY: help install run

help:
	@echo "make install   Tạo .venv và cài requirements.txt"
	@echo "make run       Chạy ứng dụng desktop (Tkinter), nạp model trực tiếp"

$(PYTHON):
	$(PY_BOOTSTRAP) -m venv $(VENV)

install: $(PYTHON)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# App tự báo lỗi nếu thiếu checkpoint.
run: $(PYTHON)
	$(PYTHON) main.py
