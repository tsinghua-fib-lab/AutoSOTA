import re

def normalize_answer(s):
    """标准化答案以便统一比较"""
    # 转换为小写
    s = s.lower()
    # 去除两端的空白字符
    s = s.strip()
    s = s.replace(',', '')
    s = s.replace('and', '')
    # 统一引号
    s = s.replace("'", "")
    s = s.replace("\"", "")
    # 将数字的文字形式转换为数值形式
    number_words = {
        r'\bone\b': '1',
        r'\btwo\b': '2',
        r'\bthree\b': '3',
        r'\bfour\b': '4',
        r'\bfive\b': '5',
        r'\bsix\b': '6',
        r'\bseven\b': '7',
        r'\beight\b': '8',
        r'\bnine\b': '9',
        r'\bten\b': '10'
    }
    for word, number in number_words.items():
        s = re.sub(word, number, s)
    s = s.replace('.', '')
    s = s.replace('–', '-')
    s = s.replace('-', '-')
    s = s.replace('*', '')
    return s

def can_convert_to_number(s):
    try:
        # 尝试将字符串转换为整数
        int(s)
        return True
    except ValueError:
        pass  # 如果转换为整数失败，尝试转换为浮点数

    try:
        # 尝试将字符串转换为浮点数
        float(s)
        return True
    except ValueError:
        # 如果转换为浮点数也失败，则字符串不能转换为数字
        return False
    
def extract_keywords_and_numbers(reference):
    keywords = []
    numbers = []

    if bool(re.search(r'\d', reference[0])):
        numbers = []
        for num in reference:
            # 先使用正则表达式匹配出所有可能的数字（包括带小数点和正负号的）
            possible_numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+', num)
            cleaned_numbers = []
            for possible_num in possible_numbers:
                # 去除数字中的非数字字符（除了小数点、正负号）
                cleaned_num = re.sub(r'[^0-9.+-]', '', possible_num)
                # 将清理后的数字添加到列表中
                cleaned_numbers.append(cleaned_num)
            # 将每个num处理后的数字列表添加到最终的列表中
            numbers += cleaned_numbers
    else:
        keywords = reference


    return keywords, numbers

def calculate_relative_error(reference_value, assistant_value):
    # 计算相对误差的函数
    if reference_value == 0:
        return float('inf')  # 避免除以零的情况
    return abs((reference_value - assistant_value) / reference_value)


def exact_match(reference_answer, assistant_answer):
    # 提取关键词和数值
    keywords, numbers = extract_keywords_and_numbers(reference_answer)

    is_match = False
    if keywords != []:
        # 检查关键词是否都在assistant_answer中
        is_match = all(keyword in assistant_answer for keyword in keywords)
    elif numbers != []:
        assistant_numbers = [float(num) for num in re.findall(r'[-+]?[0-9]*\.?[0-9]+', assistant_answer)]
        reference_numbers = [float(num) for num in numbers]

        is_match = all(any(calculate_relative_error(ref_num, assistant_num) <= 0.01 or round(abs(assistant_num)) == abs(ref_num) or (ref_num*1000 == assistant_num) or (assistant_num*1000 == ref_num) for assistant_num in assistant_numbers) for ref_num in reference_numbers)

    return is_match