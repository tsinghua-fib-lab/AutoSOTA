# evaluation methods
def compare(target, current):
    # input:
    # target: true DAG
    # current: learned DAG
    # output:
    # tp: true positive (edges appear in both target and current)
    # fp: false positive (edges appear in current but not in target)
    # fn: false negative (edges appear in target but not in current)

    compare_dict = {}
    tp = 0
    fp = 0
    fn = 0
    for key, value in target.items():
        for par in value['par']:
            if par in current[key]['par']:
                tp = tp + 1
            else:
                fn = fn + 1
        for nei in value['nei']:
            if nei in current[key]['nei']:
                tp = tp + 0.5
            else:
                fn = fn + 0.5
    for key, value in current.items():
        for par in value['par']:
            if par not in target[key]['par']:
                fp = fp + 1
        for nei in value['nei']:
            if nei not in target[key]['nei']:
                fp = fp + 0.5

    compare_dict['tp'] = tp
    compare_dict['fp'] = fp
    compare_dict['fn'] = fn
    compare_dict['f1'] = 2 * tp / (2 * tp + fp + fn)
    return compare_dict