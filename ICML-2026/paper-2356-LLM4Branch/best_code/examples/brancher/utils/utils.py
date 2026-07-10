import ecole
import numpy as np
import re
import sys
import importlib.util
import ast

DATASET_PARAMS = {
    'setcover': {
        'easy': {
            'n_rows': 500,
            'n_cols': 1000,
        },
        'medium': {
            'n_rows': 1000,
            'n_cols': 1000,
        },
        'hard': {
            'n_rows': 2000,
            'n_cols': 1000,
        }
    },
    'cauctions': {
        'easy': {
            'n_items': 100,
            "n_bids": 500,
        },
        'medium': {
            'n_items': 200,
            "n_bids": 1000,
        },
        'hard': {
            'n_items': 300,
            "n_bids": 1500,
        }
    },
    'indset': {
        'easy': {
            'n_nodes': 750,
            'affinity': 4,
        },
        'medium': {
            'n_nodes': 1000,
            'affinity': 4,
        },
        'hard': {
            'n_nodes': 1500,
            'affinity': 4,
        }
    },
    'facilities': {
        'easy': {
            'n_customers': 100, 
            'n_facilities': 100,
        },
        'medium': {
            'n_customers': 200, 
            'n_facilities': 100,
        },
        'hard': {
            'n_customers': 400, 
            'n_facilities': 100,
        }
    }
}

def load_program(program_path):
    spec = importlib.util.spec_from_file_location("score_function", program_path)
    module = importlib.util.module_from_spec(spec)

    # 加载模块到 sys.modules
    sys.modules["score_function"] = module
    spec.loader.exec_module(module)

    # 现在可以像普通导入一样访问函数了
    score_function = getattr(module, "score_function")

    used_features = []

    with open(program_path, 'r', encoding='utf-8') as f:
        program_content = f.read()

    # 匹配 'USED_FEATURES = ' 后面跟着一个列表
    match = re.search(r'USED_FEATURES\s*=\s*\[(.*?)\]', program_content)
    if match and match.group(1):
        list_str = match.group(1)
        used_features = [int(num.strip()) for num in list_str.split(',')] 
    else:
        raise ValueError("程序中'USED_FEATURES' 定义不完整。请确保程序包含类似 'USED_FEATURES = [1, 2, 3]' 的行。")
    
    match = re.search(r'PARAMS\s*=\s*\[(.*?)\]', program_content)
    if match and match.group(1):
        list_str = match.group(1)
        function_params = [float(num.strip()) for num in list_str.split(',')] 
    else:
        raise ValueError("程序中'PARAMS' 定义不完整。请确保程序包含类似 'PARAMS= [1.0, 2.0, 3.0]' 的行。")
    
    match = re.search(r'BOUNDS\s*=\s*', program_content)
    if not match:
        raise ValueError("在程序中未找到 'BOUNDS =' 的定义。")
    content_after_equals = program_content[match.end():]
    lstripped_content = content_after_equals.lstrip()
    if not lstripped_content.startswith('['):
        raise ValueError("'BOUNDS =' 后面必须紧跟着一个列表定义 '['。")
    offset = len(content_after_equals) - len(lstripped_content)
    start_index = match.end() + offset
    bracket_level = 0
    end_index = -1
    for i, char in enumerate(program_content[start_index:]):
        if char == '[':
            bracket_level += 1
        elif char == ']':
            bracket_level -= 1
        if bracket_level == 0:
            end_index = start_index + i
            break
    if end_index == -1:
        raise ValueError("程序中 'BOUNDS' 列表的括号没有正确闭合。")
    list_str = program_content[start_index : end_index + 1]
    try:
        bound_params = ast.literal_eval(list_str)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"程序中'BOUNDS' 的格式不正确: {e}")

    return score_function, used_features, function_params, bound_params

def normalize(variable_features, used_features):
    subset = variable_features[:, used_features]
    min_vals = np.min(subset, axis=0)
    max_vals = np.max(subset, axis=0)
    range_vals = max_vals - min_vals
    safe_range = np.where(range_vals > 0, range_vals, 1)
    variable_features[:, used_features] = (subset - min_vals) / safe_range
    return variable_features

def create_instance(dataset, seed, level = None, dir = None):

    if dataset == 'setcover':
        generator = ecole.instance.SetCoverGenerator(n_rows = DATASET_PARAMS[dataset][level]['n_rows'], n_cols =  DATASET_PARAMS[dataset][level]['n_cols'], density = 0.05)
    elif dataset == 'cauctions': 
        generator = ecole.instance.CombinatorialAuctionGenerator(n_items = DATASET_PARAMS[dataset][level]['n_items'], n_bids = DATASET_PARAMS[dataset][level]['n_bids'])
    elif dataset == 'indset': 
        generator = ecole.instance.IndependentSetGenerator(n_nodes = DATASET_PARAMS[dataset][level]['n_nodes'], affinity = DATASET_PARAMS[dataset][level]['affinity'], graph_type = ecole.instance.IndependentSetGenerator.GraphType.barabasi_albert)
    elif dataset == 'facilities':
        generator = ecole.instance.CapacitatedFacilityLocationGenerator(n_customers = DATASET_PARAMS[dataset][level]['n_customers'], n_facilities = DATASET_PARAMS[dataset][level]['n_facilities'], ratio = 5)
    else:
        raise ValueError("dataset is expected as setcover, cauctions, indset, facilities")
    generator.seed(seed)
    return next(generator)
    
def get_scip_params(time_limit = None, gap_limit = None):
    if time_limit:
        return {
        'separating/maxrounds': 0,
        'presolving/maxrestarts': 0,
        "branching/vanillafullstrong/idempotent": True,
        'limits/time': time_limit,
        'timing/clocktype': 1
    }
    elif gap_limit:
        return {
        'separating/maxrounds': 0,
        'presolving/maxrestarts': 0,
        "branching/vanillafullstrong/idempotent": True,
        'limits/gap': gap_limit,
        'timing/clocktype': 1
    }
    else:
        return {
            'separating/maxrounds': 0,
            'presolving/maxrestarts': 0,
            "branching/vanillafullstrong/idempotent": True,
            'limits/time': 3600,
            'timing/clocktype': 1
        }

def geometric_mean_stable(data, shift = 1):

    a = np.asarray(data)

    if a.size == 0 or np.any(a < 0):
        return 0.0

    return float(np.exp(np.mean(np.log(a + shift)))) - shift

def numerical_mean_stable(arr):
    if len(arr) == 0:
        return 0.0
    return sum(arr) / len(arr)

def format_list(number_list):
    return [round(t, 2) for t in number_list]