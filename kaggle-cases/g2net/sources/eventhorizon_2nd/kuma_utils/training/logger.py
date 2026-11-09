import time
from pprint import pprint, pformat


def get_time(time_format='%H:%M:%S'):
    return time.strftime(time_format, time.gmtime())


class LGBMLogger:

    def __init__(self, path, stdout=True, file=False):
        self.path = path
        self.stdout = stdout
        self.file = file
        log_str = f'Logger created at {get_time("%y/%m/%d:%H:%M:%S")}'
        if self.stdout:
            print(log_str)
        if self.file:
            with open(self.path, 'w') as f:
                f.write(log_str + '\n')
      
    def lgbm(self, env):
        log_str = f'{get_time()} '
        log_str += f'[iter {env.iteration:-5}] '
        for inputs in env.evaluation_result_list:
            for i in inputs:
                if isinstance(i, str):
                    log_str += f'{i} '
                elif isinstance(i, bool):
                    pass
                else:
                    log_str += f'{i:.6f} '
        else:
            log_str += '/ '
        log_str += '\n'
        # if self.stdout:
        #     print(log_str)
        if self.file:
            with open(self.path, 'a') as f:
                f.write(log_str)

    def optuna(self, study, trial):
        best_score = study.best_value
        curr_score = trial.value
        if curr_score == best_score:
            log_str = f'{get_time()} '
            log_str += f'[trial {trial.number:-4}] New best: {best_score:.6f} \n'
            log_str += f'{pformat(study.best_params, compact=True, indent=2)}'
            if self.stdout:
                print(log_str)
            if self.file:
                with open(self.path, 'a') as f:
                    f.write(log_str + '\n')
        
    def __call__(self, log_str):
        log_str = get_time() + ' ' + log_str
        if self.stdout:
            print(log_str)
        if self.file:
            with open(self.path, 'a') as f:
                f.write(log_str + '\n')
