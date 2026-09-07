.PHONY: dev backend lint test eval bench benchmarks clean

dev:
	npm run dev --prefix frontend

backend:
	cd vanirakshak-backend && python run.py

lint:
	npm run lint --prefix frontend

test:
	pytest -v

eval:
	cd vanirakshak-backend && python -m vanirakshak eval

bench:
	cd vanirakshak-backend && python -m vanirakshak bench

benchmarks:
	python benchmarks/run_benchmarks.py

clean:
	python -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.pytest_cache')]"
