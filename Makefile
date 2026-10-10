VENV   := .venv
PYTHON := $(VENV)/bin/python
PORT   ?= 8501

.PHONY: start install help

help:
	@echo "make install   Tạo .venv và cài requirements.txt"
	@echo "make start     Chạy app demo Streamlit (PORT=$(PORT))"

$(PYTHON):
	python3 -m venv $(VENV)

install: $(PYTHON)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

start: $(PYTHON)
	@test -f models/transformer-scratch-vihsd/model_state_dict.pt || \
		{ echo "Thiếu checkpoint models/transformer-scratch-vihsd/ — hãy chạy notebook from-scratch trước."; exit 1; }
	$(PYTHON) -m streamlit run app/app.py --server.port $(PORT)
