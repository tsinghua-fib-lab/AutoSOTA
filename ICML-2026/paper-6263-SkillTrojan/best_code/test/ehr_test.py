import csv
import ast

def calculate_accuracy(file_path, column_name='correct', filter_empty_query=True):
    """
    读取CSV文件并计算指定列的准确率
    
    参数:
        file_path: CSV文件路径
        column_name: 要计算的列名，默认为'correct'
        filter_empty_query: 是否过滤query_result为空数组的行，默认为True
    
    返回:
        准确率(百分比)
    """
    true_count = 0
    total_count = 0
    filtered_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # 如果需要过滤空query_result
            if filter_empty_query and 'query_result' in row:
                query_result = row['query_result'].strip()
                
                # 检查是否为空数组（更严格的检查）
                try:
                    # 尝试解析为Python对象
                    parsed_result = ast.literal_eval(query_result)
                    if isinstance(parsed_result, list) and len(parsed_result) == 0:
                        filtered_count += 1
                        continue
                except:
                    # 如果解析失败，用简单的字符串比较
                    if query_result == '[]' or query_result == '':
                        filtered_count += 1
                        continue
            
            value = row[column_name].strip().upper()
            total_count += 1
            
            if value == 'TRUE':
                true_count += 1
    
    if total_count == 0:
        print("警告: 过滤后没有有效数据")
        return 0
    
    accuracy = (true_count / total_count) * 100
    
    print(f"被过滤的行数 (query_result为[]): {filtered_count}")
    print(f"有效总数: {total_count}")
    print(f"正确数 (TRUE): {true_count}")
    print(f"错误数 (FALSE): {total_count - true_count}")
    print(f"准确率: {accuracy:.2f}%")
    
    return accuracy

# 使用示例
if __name__ == "__main__":
    file_path = "ehr_outputs/ehr_results.csv"
    accuracy = calculate_accuracy(file_path, filter_empty_query=True)
