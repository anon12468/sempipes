nohup bash -lc 'poetry run python -u experiments/sigmod/rutgers/execute_sempipes_medium2_optimized.py > experiments/sigmod/rutgers/medium_optimized_3.1_pro_epsilon.log 2>&1 &
echo $! > experiments/sigmod/rutgers/medium_optimized.pid
