from mmengine.hooks import Hook

from xtuner.registry import BUILDER


class SpecialDatasetInfoHook(Hook):

    def __init__(self, tokenizer, is_intern_repo_dataset=False, special_tokens=None):
        self.tokenizer = BUILDER.build(tokenizer)
        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)
        self.is_intern_repo_dataset = is_intern_repo_dataset

    def log(self, runner, dataset, mode='train'):

        def _log(input_ids, log_prefix=''):
            if self.is_intern_repo_dataset:
                input_ids = [abs(x) for x in input_ids]

            text = self.tokenizer.decode(input_ids)
            runner.logger.info(text)

        runner.logger.info(f'Num {mode} samples {len(dataset)}')
        runner.logger.info(f'{mode} example:')
        if 'chosen_ids' in dataset[0]:
            _log(dataset[0]['chosen_ids'], log_prefix='chosen: ')
            _log(dataset[0]['rejected_ids'], log_prefix='rejected: ')
        else:
            _log(dataset[0]['input_ids'])

    def before_train(self, runner) -> None:
        do_train = runner.train_loop is not None
        do_eval = runner.val_loop is not None
        if do_train:
            train_dataset = runner.train_dataloader.dataset
            self.log(runner, train_dataset, mode='train')
        if do_eval:
            eval_dataset = runner.val_dataloader.dataset
            self.log(runner, eval_dataset, mode='eval')

    def before_val(self, runner) -> None:
        eval_dataset = runner.val_dataloader.dataset
        self.log(runner, eval_dataset, mode='eval')

    def before_test(self, runner) -> None:
        test_dataset = runner.test_dataloader.dataset
        self.log(runner, test_dataset, mode='test')
