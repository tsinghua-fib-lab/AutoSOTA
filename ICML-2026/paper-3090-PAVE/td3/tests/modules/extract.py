import os
import pandas as pd
import shutil
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

taskname = "reacher"

tasknames = ['cartpole', 'reacher', 'cheetah', 'walker']

def extract_tensorboard_logs_by_folder(log_root_dir, output_root_folder="tensorboard_extracted_logs", output_csv=True):
    """
    각 TensorBoard 로그 폴더별로 데이터를 추출하여 저장.
    :param log_root_dir: TensorBoard 로그의 최상위 폴더 (여러 개의 실험 폴더가 포함됨)
    :param output_root_folder: 로그 데이터를 저장할 폴더
    :param output_csv: CSV 저장 여부 (기본값: True)
    :return: 모든 데이터를 포함한 Pandas DataFrame
    """
    if not os.path.isdir(log_root_dir):
        return

    all_global_data = []  # 전체 데이터를 저장할 리스트
    os.makedirs(output_root_folder, exist_ok=True)  # ✅ 최상위 저장 폴더 생성

    # ✅ 최상위 폴더 내 각 실험 폴더 탐색
    for experiment_folder in os.listdir(log_root_dir):
        experiment_path = os.path.join(log_root_dir, experiment_folder)
        if not os.path.isdir(experiment_path):  # 디렉토리인지 확인
            continue
        
        print(f"🔍 처리 중: {experiment_folder}")
        experiment_output_folder = os.path.join(output_root_folder, experiment_folder)
        # os.makedirs(experiment_output_folder, exist_ok=True)  # ✅ 실험별 저장 폴더 생성

        # ✅ 해당 폴더 내 모든 TensorBoard 로그 파일 탐색
        log_files = []
        for root, _, files in os.walk(experiment_path):
            for file in files:
                if "tfevents" in file:
                    log_files.append(os.path.join(root, file))

        print(f"📂 발견된 TensorBoard 로그 파일 ({experiment_folder}): {len(log_files)}개")

        # ✅ 모든 로그 파일 처리
        subnum = 0
        for event_path in log_files:
            all_experiment_data = []  # 현재 폴더 내 모든 데이터를 저장할 리스트
            experiment_output_folder_sub = f"{experiment_output_folder}_{subnum}"
            os.makedirs(experiment_output_folder_sub, exist_ok=True)
            subnum += 1
            print(f"📂 처리 중: {event_path}")
            event_acc = EventAccumulator(event_path)
            event_acc.Reload()

            # 사용 가능한 모든 태그 가져오기
            tags = event_acc.Tags()["scalars"]
            print(f"✅ {len(tags)}개 태그 발견: {tags}")

            # 각 태그별 데이터 추출
            for tag in tags:
                scalars = event_acc.Scalars(tag)
                df = pd.DataFrame([(s.step, s.value) for s in scalars], columns=["Step", "Value"])
                df["Tag"] = tag
                df["Source"] = os.path.basename(event_path)  # 파일 경로 대신 파일 이름만 저장
                all_experiment_data.append(df)

                # ✅ 실험별 태그 데이터 저장 (Append 모드)
                tag_filename = f"{experiment_output_folder_sub}/log_data_{tag.replace('/', '_')}.csv"
                if output_csv:
                    if os.path.exists(tag_filename):
                        df.to_csv(tag_filename, mode="a", header=False, index=False)
                    else:
                        df.to_csv(tag_filename, index=False)

            # ✅ 실험별 모든 데이터를 하나의 CSV로 저장
            if all_experiment_data:
                final_experiment_df = pd.concat(all_experiment_data, ignore_index=True).drop_duplicates()
                all_global_data.append(final_experiment_df)  # 글로벌 데이터에 추가
                experiment_csv_path = os.path.join(experiment_output_folder_sub, "all_tensorboard_data.csv")

                if output_csv:
                    final_experiment_df.to_csv(experiment_csv_path, index=False)
                    print(f"✅ {experiment_folder}의 모든 데이터가 {experiment_csv_path}로 저장됨")

    # ✅ 모든 실험 데이터를 하나의 CSV로 저장
    if all_global_data:
        final_global_df = pd.concat(all_global_data, ignore_index=True).drop_duplicates()
        global_csv_path = os.path.join(output_root_folder, "global_tensorboard_data.csv")

        if output_csv:
            final_global_df.to_csv(global_csv_path, index=False)
            print(f"✅ 모든 실험의 데이터를 {global_csv_path}로 저장됨")

    return final_global_df


def extract_files(taskname, rootdir = ""):
    # ✅ 실행 코드
    log_root_directory = f"{rootdir}tensorboard_logs/{taskname}"  # 🚀 최상위 TensorBoard 로그 폴더 경로 입력
    output_dir = f"{rootdir}temp_logs/tensorboard_extracted_logs_{taskname}"
    df = extract_tensorboard_logs_by_folder(log_root_directory, output_dir)

def extract_second_column_from_csv(input_folder, filtered_folder, allist=["Lips_L_TD3_", "Vanilla_"]):
    if not os.path.exists(input_folder):
        print(f"❌ 경로가 존재하지 않음: {input_folder}")
        return
    
    os.makedirs(filtered_folder, exist_ok=True)
    
    find_csv_list = ["log_data_rollout_ep_rew_mean.csv", "log_data_train_oscillation.csv"]
    dtframes = {}
    for csv_name in find_csv_list:
        dtframes[csv_name] = pd.DataFrame()
    for alname in allist:
        column_dict = {}
        for csv_name in find_csv_list:
            column_dict[csv_name] = []
        for dirname in os.listdir(input_folder):
            dirpath = os.path.join(input_folder, dirname)
            if not os.path.isdir(dirpath):  # 디렉토리인지 확인
                continue
            if not dirname.startswith(alname):
                continue
            for filename in os.listdir(dirpath):
                if filename in find_csv_list:
                    filepath = os.path.join(dirpath,filename)
                    try:
                        # CSV 파일 읽기
                        df = pd.read_csv(filepath)

                        # 두 번째 열이 존재하는지 확인
                        if df.shape[1] < 2:
                            print(f"⚠️ {filename}: 두 번째 열이 없음, 건너뜀.")
                            continue

                        # 두 번째 열만 추출
                        second_column_df = df.iloc[:, [1]]
                        
                        column_dict[filename].append(second_column_df)

                    except Exception as e:
                        print(f"❌ {filename} 처리 중 오류 발생: {e}")
        # print(column_dict)
        try:
            for csv_name in find_csv_list:
                series_list = column_dict[csv_name]
                if not series_list:
                    print(f"⚠️ {alname} 에 {csv_name} 데이터가 하나도 없음, 건너뜀.")
                    continue
                
                    # 각 실행마다 마지막 행 값만 뽑아서 리스트로
                last_values = [s.iloc[-1] for s in series_list]
                last_avg    = float(pd.Series(last_values).mean())

                # 결과 DataFrame 에 column 추가 (한 행짜리)
                dtframes[csv_name].loc[0, alname] = last_avg

        except Exception as e:
            print(f"오류 발생: {e}")

    for csv_name in find_csv_list:
        new_filepath = os.path.join(filtered_folder, csv_name)
        dtframes[csv_name].to_csv(new_filepath, index=False)


def extract_csv(taskname, rootdir = "", allist=["Lips_L_TD3_", "Vanilla_"]):
    csv_folder_path = f"{rootdir}temp_logs/tensorboard_extracted_logs_{taskname}" 
    filtered_folder_path = f"{rootdir}filtered/filtered_{taskname}"
    extract_second_column_from_csv(csv_folder_path, filtered_folder_path, allist)


def extract_all(rootdir = "", allist=["Lips_L_TD3_", "Vanilla_"]):
    for t in tasknames:
        extract_files(t, rootdir)

    for t in tasknames:
        extract_csv(t, rootdir, allist)
    
    temp_dir = f"{rootdir}temp_logs/"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)