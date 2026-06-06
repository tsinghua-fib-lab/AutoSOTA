import numpy as np

try:
    from catenets.models.jax import RANet, SNet1, SNet2, SNet, FlexTENet, XNet
except ImportError:
    print("catenets not installed, skipping CATENet baselines.")

from sklearn.preprocessing import StandardScaler

from .base import BaselineModel

# parameter grid for HPO
PARAM_GRID = {
    "n_layers_r": [2, 3],  # representation depth
    "n_units_r": [128, 256],  # representation width
    "n_layers_out": [1, 2],  # head depth
    "n_units_out": [128, 256],  # head width
}


class CATENetBaseline(BaselineModel):
    def __init__(self, model, hpo: bool = True):
        super().__init__(hpo)
        self.model = model

    def _scale_x(self, X_tr, X_te):
        scaler_x = StandardScaler().fit(X_tr)
        return (scaler_x.transform(X_tr).astype("float32"), scaler_x.transform(X_te).astype("float32"), scaler_x)

    def _scale_y(self, y_tr):
        scaler_y = StandardScaler().fit(y_tr.reshape(-1, 1))
        y_tr_s = scaler_y.transform(y_tr.reshape(-1, 1)).ravel().astype("float32")
        return y_tr_s, scaler_y

    def _unscale_cate(self, cate_s, scaler_y):
        return cate_s * scaler_y.scale_[0] if scaler_y is not None else cate_s

    def estimate_cate(self, X_train, t_train, y_train, X_test):
        # — train split
        X_tr = np.asarray(X_train)
        y_tr = np.asarray(y_train).ravel()
        w_tr = np.asarray(t_train).ravel()
        X_te = np.asarray(X_test)

        # — scale
        X_tr_s, X_te_s, _ = self._scale_x(X_tr, X_te)
        y_tr_s, scaler_y = self._scale_y(y_tr)

        try:
            if self.hpo:
                self.model.fit_and_select_params(
                    X=X_tr_s,
                    y=y_tr_s,
                    w=w_tr,
                    param_grid=PARAM_GRID,
                )
            else:
                self.model.fit(
                    X=X_tr_s,
                    y=y_tr_s,
                    w=w_tr,
                )
            cate_pred_s = self.model.predict(X_te_s).ravel()
            cate_pred = self._unscale_cate(cate_pred_s, scaler_y)
        except Exception as e:
            print(e)
            cate_pred = np.zeros(X_te.shape[0], dtype=np.float32)
        return cate_pred

    def estimate_ate(self, X, t, y):
        pred_cate = self.estimate_cate(X, t, y, X)
        return np.mean(pred_cate)


class RANetBaseline(CATENetBaseline):
    def __init__(self, hpo: bool = True, batch_size: int = 512):
        model = RANet(batch_size=batch_size)
        super().__init__(model, hpo)


class TarNetBaseline(CATENetBaseline):
    def __init__(self, hpo: bool = True, batch_size: int = 512):
        model = SNet1(batch_size=batch_size)
        super().__init__(model, hpo)


class DragonNetBaseline(CATENetBaseline):
    def __init__(self, hpo: bool = True, batch_size: int = 512):
        model = SNet2(batch_size=batch_size)
        super().__init__(model, hpo)


class SNetBaseline(CATENetBaseline):
    def __init__(self, hpo: bool = True, batch_size: int = 512):
        model = SNet(batch_size=batch_size)
        super().__init__(model, hpo)


class FlexTENetBaseline(CATENetBaseline):
    def __init__(self, hpo: bool = True, batch_size: int = 512):
        model = FlexTENet(batch_size=batch_size)
        super().__init__(model, hpo)


class XNetBaseline(CATENetBaseline):
    def __init__(self, hpo: bool = True, batch_size: int = 512):
        model = XNet(batch_size=batch_size)
        super().__init__(model, hpo)
