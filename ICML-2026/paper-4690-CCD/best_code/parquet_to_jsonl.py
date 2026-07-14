import pandas as pd

def convert_parquet_to_jsonl(parquet_path, jsonl_path):
    # 1. Parquet 파일 읽기
    df = pd.read_parquet(parquet_path)
    
    # 2. JSONL 형식으로 저장 (orient='records', lines=True 옵션 사용)
    # force_ascii=False는 한글이나 특수문자가 깨지지 않게 해줍니다.
    df.to_json(jsonl_path, orient='records', lines=True, force_ascii=False)
    
    print(f"변환 완료: {parquet_path} -> {jsonl_path}")
    print(f"총 {len(df)}개의 데이터가 저장되었습니다.")

# 실험에 필요한 두 파일을 각각 변환해보세요.
if __name__ == "__main__":
    # 마스크 정보가 담긴 파일
    #convert_parquet_to_jsonl('templates/metadata.parquet', 'templates_metadata.jsonl')
    
    
    # 229개 프롬프트와 시드 정보가 담긴 파일
    # name = "sdv2_bb_attack_gt_verify.parquet"
    # convert_parquet_to_jsonl(name, name.replace(".parquet", ".jsonl"))

    import argparse
    import sys
    import os

    parser = argparse.ArgumentParser(description='Convert parquet to jsonl')
    parser.add_argument('--input_file', type=str, help='Input parquet file path')
    args = parser.parse_args()

    if args.input_file:
        name = args.input_file
    else:
        name = "sdv2_bb_attack_gt_verify.parquet"

    if os.path.exists(name):
        jsonl_name = name.replace(".parquet", ".jsonl")
        convert_parquet_to_jsonl(name, jsonl_name)
        
        # Filter for TV cases
        tv_output_name = jsonl_name.replace(".jsonl", "_TV.jsonl")
        print(f"Filtering {jsonl_name} for 'TV' cases...")
        df = pd.read_json(jsonl_name, lines=True)
        if 'overfit_type' in df.columns:
            df_tv = df[df['overfit_type'] == 'TV']
            df_tv.to_json(tv_output_name, orient='records', lines=True, force_ascii=False)
            print(f"Saved {len(df_tv)} TV cases to {tv_output_name}")
            
            # Filter for MV/RV cases
            mvrv_output_name = jsonl_name.replace(".jsonl", "_MVRV.jsonl")
            df_mvrv = df[df['overfit_type'].isin(['MV', 'RV', 'MVRV'])]
            if len(df_mvrv) > 0:
                df_mvrv.to_json(mvrv_output_name, orient='records', lines=True, force_ascii=False)
                print(f"Saved {len(df_mvrv)} MVRV cases to {mvrv_output_name}")
        else:
            print("Column 'overfit_type' not found in data.")
    else:
        #git test
        print("Usage: python parquet_to_jsonl.py --input_file <input_parquet_file>") 