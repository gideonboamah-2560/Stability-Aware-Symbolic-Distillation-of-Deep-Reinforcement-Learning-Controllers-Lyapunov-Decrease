for s in 3 4; do [ -f models/sac_seed$s.zip ] || python3 train_teacher.py $s; done
python3 experiment.py && python3 test_stage.py && echo ALLDONE
