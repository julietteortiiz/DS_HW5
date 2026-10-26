census:
	python3 run_NB.py -r data/1994_census_cleaned_train.csv -e data/1994_census_cleaned_test.csv

zoo:
	python3 run_NB.py -r data/zoo_train.arff -e data/zoo_test.arff