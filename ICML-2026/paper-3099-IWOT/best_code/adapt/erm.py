class ERM:
    def __init__(self, model, fabric, loss_fun, opt):
        self.name = "ERM"
        print(f"Initializing {self.name}")
        self.loss_fun = loss_fun
        self.opt = opt
        model, self.opt = fabric.setup(model, self.opt)

    def adapt(self, model, fabric, X_source, y_source, X_target, y_target=[]):
        pred_source = model(X_source)
        loss = self.loss_fun(pred_source, y_source)
        fabric.backward(loss)
        self.opt.step()
        self.opt.zero_grad()
