.PHONY: setup dev backend lint test integration-check check-docs eval bench benchmarks clean

# Installs everything `make test` needs (15 tests) plus the real ECAPA-TDNN /
# WavLM detector backends and the trained ml/ identity model -- see
# requirements-dev.txt for exactly what needs what and why.
setup:
	pip install -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cpu

dev:
	npm run dev --prefix frontend

backend:
	cd vanirakshak-backend && python run.py

lint:
	npm run lint --prefix frontend

test:
	pytest -v

integration-check:
	python scripts/integration_check.py

check-docs:
	python scripts/check_docs.py

eval:
	cd vanirakshak-backend && python -m vanirakshak eval

bench:
	cd vanirakshak-backend && python -m vanirakshak bench

benchmarks:
	python benchmarks/run_benchmarks.py

clean:
	python -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.pytest_cache')]"
