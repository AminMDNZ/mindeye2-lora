CONFIG ?= configs/colab_t4.yaml
PY     ?= python -m mindeye_lora.cli

.PHONY: setup assets verify precompute train predict recon evaluate compare report all smoke test status clean-runs

setup:      ; $(PY) setup      --config $(CONFIG)
assets:     ; $(PY) assets     --config $(CONFIG)
verify:     ; $(PY) verify     --config $(CONFIG)
precompute: ; $(PY) precompute --config $(CONFIG)
train:      ; $(PY) train      --config $(CONFIG)
predict:    ; $(PY) predict    --config $(CONFIG)
recon:      ; $(PY) recon      --config $(CONFIG)
evaluate:   ; $(PY) evaluate   --config $(CONFIG)
compare:    ; $(PY) compare    --config $(CONFIG)
report:     ; $(PY) report     --config $(CONFIG)
status:     ; $(PY) status     --config $(CONFIG)
all:        ; $(PY) run-all    --config $(CONFIG)
smoke:      ; $(PY) run-all    --config configs/smoke.yaml

test:
	pytest tests/ -q

clean-runs:
	@echo "This deletes training runs from the workspace but keeps downloaded assets."
	rm -rf $${MINDEYE_LORA_ROOT:-workspace}/runs $${MINDEYE_LORA_ROOT:-workspace}/results
