import torch
import time
from competitor_methods.pfns import RiemannDistribution
from tabpfn import TabPFNRegressor  


def construct_tabpfn_inputs(
    phi, x, z
):
    batch_size = x.shape[0] if len(x.shape) > 0 else 1

    phi_keys_sorted = sorted([key for key in phi.keys()])
    obs_keys_sorted = sorted([key for key in z.keys()])

    phi_stacked = torch.cat([phi[key].reshape(batch_size, -1) for key in phi_keys_sorted], dim=-1)
    z_stacked = torch.cat([z[key].reshape(batch_size, -1) for key in obs_keys_sorted], dim=-1)
    x = x.reshape(batch_size, -1)

    inp = torch.cat([phi_stacked, z_stacked], dim=-1)
    output = x

    return inp, output

def test_tabpfn(
    phi_dict_test, x_test, z_test,
    complete_distribution,
    tabpfn_trainsize,
    cap_logits=-100.0
    ):

    # PFN-type models can only handle univariate outputs, hence event shape = ()
    batch_size = x_test.shape
    device = z_test[list(z_test.keys())[0]].device
    
    phi_train, x_train, z_train = complete_distribution.sample((tabpfn_trainsize,))
    phi_prior_dict = complete_distribution.meta_prior.decode_sample(phi_train)

    X_train, Y_train = construct_tabpfn_inputs(phi_prior_dict, x_train, z_train)
    X_test, Y_test = construct_tabpfn_inputs(phi_dict_test, x_test, z_test)

    regressor = TabPFNRegressor()

    regressor.fit(X_train.cpu(), Y_train.cpu())
    
    prediction_start_time = time.time()

    regressor_prediction = regressor.predict(X_test.cpu(), output_type="full")
    
    print("TabPFN prediction time:", time.time() - prediction_start_time)
    
    logits = regressor_prediction['logits']
    criterion = regressor_prediction['criterion']
    
    logits = torch.clamp(logits, min=cap_logits)
     
    output_distribution = RiemannDistribution(
        probs=logits.softmax(dim=-1).reshape(batch_size + (-1,)).to(device), 
        borders=criterion.borders.to(device),
    )

    nll = -output_distribution.log_prob(Y_test.swapdims(0, -1).to(device))

    return output_distribution, nll, regressor

