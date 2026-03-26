from environment import *


class LRScheduler:
    def __init__(self, CFG):
        self.CFG = CFG

    def __call__(self, optimizer, epoch):
        if self.CFG.lr_scheduler == 'type1':
            lr = {epoch: self.CFG.lr * (0.5 ** ((epoch - 1) // 1))}
        elif self.CFG.lr_scheduler == 'type2':
            lr = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
        elif self.CFG.lr_scheduler == 'cosine':
            lr = {epoch: self.CFG.lr / 2 * (1 + math.cos(epoch / self.CFG.epochs * math.pi))}
        else:
            lr = {}
        if epoch in lr.keys():
            new_lr = lr[epoch]
            for param_group in optimizer.param_groups:
                old_lr = param_group['lr']
                param_group['lr'] = new_lr
            print(f"[LR UPDATED] {old_lr} -> {new_lr}")


class StyleColumn(progress.ProgressColumn):
    def __init__(self, column, new_style=''):
        super().__init__()
        self.column = column
        self.new_style = new_style

    def render(self, task):
        text = self.column.render(task)
        if self.new_style:
            text.stylize(self.new_style)
        else:
            text.style = ''
        return text


class ProgressBar:
    def __init__(self, CFG, description, total):
        self.CFG = CFG
        self.console = console.Console(force_terminal=True, width=200)
        self.progress = progress.Progress(
            progress.TextColumn("{task.percentage:>3.0f}%"),
            progress.BarColumn(bar_width=50, style='white', complete_style='red', finished_style='green', pulse_style='blue'),
            StyleColumn(progress.MofNCompleteColumn()),
            progress.SpinnerColumn(style='', finished_text='•'),
            progress.TextColumn("["),
            StyleColumn(progress.TimeElapsedColumn()),
            progress.TextColumn("<"),
            StyleColumn(progress.TimeRemainingColumn()),
            progress.TextColumn(", {task.fields[speed]:#.4g} ms/it , {task.fields[memory]:#.4g} GB ]  - "),
            progress.TextColumn("Loss: {task.fields[loss]:>6.3f}, "),
            progress.TextColumn("{task.fields[metric1_name]}: {task.fields[metric1]:>6.3f}, "),
            progress.TextColumn("{task.fields[metric2_name]}: {task.fields[metric2]:>6.3f}"),
            console=self.console
        )
        self.task = self.progress.add_task(
            description,
            total=total,
            speed=0.0,
            memory=0.0,
            loss=0.0,
            metric1=0.0,
            metric2=0.0,
            metric1_name=self.CFG.metric1,
            metric2_name=self.CFG.metric2
        )

    def __call__(self, memory, loss, metric1, metric2):
        speed = self.progress.tasks[self.task].speed
        self.progress.update(
            self.task,
            advance=1,
            speed=1000 / speed if speed else 0.0,
            memory=memory / (1024 ** 3),
            loss=np.mean(loss),
            metric1=np.mean(metric1),
            metric2=np.mean(metric2)
        )


class AttnVisualizer:
    def __init__(self, CFG):
        self.CFG = CFG

    def __call__(self, *args, **kwargs):
        assert False


class TestVisualizer:
    def __init__(self, CFG):
        self.CFG = CFG

    def __call__(self, pred, true, name):
        path = os.path.join(
            self.CFG.root_path,
            'plots',
            self.CFG.model,
            self.CFG.data_path.split('/')[1],
            f"{self.CFG.data_path.split('/')[-1].split('.')[0]}_{self.CFG.pred_len}"
        )
        if not os.path.exists(path):
            os.makedirs(path)
        path = f"{path}/{name}.pdf"
        plt.figure(figsize=(9, 4))
        plt.plot(pred, label='pred', linewidth=2)
        plt.plot(true, label='true', linewidth=2)
        plt.legend()
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()
