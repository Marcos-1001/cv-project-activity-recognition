.PHONY: feature_extraction model both

feature_extraction:
	python3 feature_extraction.py

model:
	python3 model.py

both:
	$(MAKE) feature_extraction
	$(MAKE) model