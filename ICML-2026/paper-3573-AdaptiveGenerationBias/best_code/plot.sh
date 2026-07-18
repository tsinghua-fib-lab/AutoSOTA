python visualization/multi_fitness_plots.py --attr_paths "gender:cab/gender, race:cab/race, religion:cab/religion" --plot_diff --compare_attr_paths "gender:cab_implicit/gender, race:cab_implicit/race, religion:cab_implicit/religion"

python visualization/domain_wordcloud.py --attr_paths "gender:cab/gender, race:cab/race, religion:cab/religion" --domain_map_json "cab_domain.json" --superdomain_map_json "cab_superdomains.json" --output_dir "plots" --domain_col "domain" --superdomain_col "superdomain" --top_n 300

python visualization/qa_length_stats.py   --attr_paths "gender:cab/gender, race:cab/race, religion:cab/religion"   --output_dir plots

python visualization/bias_categories.py --run_path cab/gender --categories_file cab/metadata/typical_gender.json --output_dir reports --bias_attribute gender
python visualization/bias_categories.py --run_path cab/race --categories_file cab/metadata/typical_race.json --output_dir reports --bias_attribute race
python visualization/bias_categories.py --run_path cab/religion --categories_file cab/metadata/typical_religion.json --output_dir reports --bias_attribute religion

python visualization/simple_fitness_plot.py --run_path cab/gender --output_dir plots --bias_attribute gender
python visualization/simple_fitness_plot.py --run_path cab/race --output_dir plots --bias_attribute race
python visualization/simple_fitness_plot.py --run_path cab/religion --output_dir plots --bias_attribute religion

python visualization/plot_fitness_by_judge.py --run_path cab/gender/model_rejudge