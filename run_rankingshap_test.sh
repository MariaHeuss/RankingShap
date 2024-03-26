mkdir results
mkdir results/background_data_files
mkdir results/model_files
mkdir results/results_MQ2008
mkdir results/results_MQ2008/feature_attributes
mkdir results/results_MSLR-WEB10K
mkdir results/results_MSLR-WEB10K/feature_attributes

python3 train_model.py --file_name test_model --dataset MQ2008 --model_type listwise
python3 generate_feature_attribution_explanations.py --model_file test_model --dataset MQ2008 --experiment_iteration 1 --test
python3 estimate_ground_truth_attribution_values.py --model_file test_model --dataset MQ2008 --background_samples 100 --nsamples 10 --experiment_iterations 3 --test
python3 evaluate_feature_attribution_with_ground_truth.py --dataset MQ2008 --file_name_ground_truth feature_importance_approxiation__backgroundcount_100_nsamples_1024_means_test.csv --test
