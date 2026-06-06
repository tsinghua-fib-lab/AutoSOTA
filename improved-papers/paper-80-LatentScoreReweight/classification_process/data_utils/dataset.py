from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np

class Weighted_Dataset(Dataset):
    def __init__(self, args, latent_z, y, dataname, returned_attr,
                use_weight=False, weight_criterion=None, temperature=0.1, 
                weight_store_path=None, return_index=False):
        
        self.dataname = dataname
        self.x = latent_z
        y = y.reshape(-1)
        if dataname == "adult":
            y[y == " >50K"] = 1
            y[y == " <=50K"] = 0
        elif dataname == "shoppers":
            y[y == False] = 0
            y[y == True] = 1
        elif dataname == "default":
            pass
        elif dataname == "bank":
            y[y == "no"] = 0
            y[y == "yes"] = 1
        elif dataname[:10] == "ACS_income":
            pass
        self.y = y

        self.use_weight = use_weight
        if use_weight:
            
            if weight_criterion == "common_error":
                unweight_error = torch.load(f"./weights/{dataname}/unweight_error_2.pt")
                sigma = torch.load(f"./weights/{dataname}/sigma_2.pt")
                
                if args.timestep_weight_criterion == "up":
                    time_weight = sigma
                elif args.timestep_weight_criterion == "down":
                    time_weight = 1 / sigma ** 2
                elif args.timestep_weight_criterion == "EDM":
                    sigma_data = 0.5
                    time_weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
                elif args.timestep_weight_criterion == "avg":
                    time_weight = torch.ones_like(sigma).to(sigma.device)
                elif args.timestep_weight_criterion == "updown":
                    P_mean = -1.2
                    a = 1
                    # time_weight = 1 / (sigma - torch.exp(torch.tensor(P_mean))) ** 2
                    time_weight = - a * (sigma - torch.exp(torch.tensor(P_mean))) ** 2
                    if torch.min(time_weight) < 0:
                        time_weight -= torch.min(time_weight)# make sure all weight positive
                elif args.timestep_weight_criterion == "downup":
                    P_mean = -1.2
                    time_weight = (sigma - torch.exp(torch.tensor(P_mean))) ** 2

                
                time_weight = time_weight / torch.max(time_weight)
                time_weight = time_weight.unsqueeze(1)
                error = time_weight * unweight_error

                mean_error = torch.mean(error, dim=(1, 2))


                if args.error_reflection == "linear":
                    weight = mean_error
                elif args.error_reflection == "softmax":
                    self.temperature = temperature
                    weight = torch.exp(mean_error / self.temperature)
                self.normalized_weight = weight / torch.sum(weight) * self.x.shape[0]

                df = pd.DataFrame(self.normalized_weight.numpy(), columns=["weight"])
                df.to_csv(f"{weight_store_path}/weight.csv")
            
            elif weight_criterion == "error_diff":
                unweight_zero_model_error = torch.load(f"./weights/{dataname}/unweight_error_0.pt") # .mean(dim=(1, 2))
                unweight_one_model_error = torch.load(f"./weights/{dataname}/unweight_error_1.pt") # .mean(dim=(1,2))

                zero_model_sigma = torch.load(f"./weights/{dataname}/sigma_0.pt")
                one_model_sigma = torch.load(f"./weights/{dataname}/sigma_1.pt")

                if args.timestep_weight_criterion == "down":
                    zero_model_time_weight = 1 / zero_model_sigma ** 2
                    one_model_time_weight = 1 / one_model_sigma ** 2
                elif args.timestep_weight_criterion == "avg":
                    zero_model_time_weight = torch.ones_like(zero_model_sigma)
                    one_model_time_weight = torch.ones_like(one_model_sigma)
                elif args.timestep_weight_criterion == "EDM":
                    sigma_data = 0.5
                    zero_model_time_weight = (zero_model_sigma ** 2 + sigma_data ** 2) / (zero_model_sigma * sigma_data) ** 2
                    one_model_time_weight = (one_model_sigma ** 2 + sigma_data ** 2) / (one_model_sigma * sigma_data) ** 2

                # zero_model_time_weight = zero_model_time_weight / torch.max(zero_model_time_weight)
                zero_model_time_weight = zero_model_time_weight.unsqueeze(1)
                zero_model_error = zero_model_time_weight * unweight_zero_model_error
                zero_model_error = torch.mean(zero_model_error, dim=(1, 2))

                # one_model_time_weight = one_model_time_weight / torch.max(one_model_time_weight)
                one_model_time_weight = one_model_time_weight.unsqueeze(1)
                one_model_error = one_model_time_weight * unweight_one_model_error
                one_model_error = torch.mean(one_model_error, dim=(1, 2))

                zero_class_error_diff = one_model_error - zero_model_error
                one_class_error_diff = zero_model_error - one_model_error
                error_diff = np.zeros(self.y.shape[0])
                error_diff[y == 0] = zero_class_error_diff[y == 0]
                error_diff[y == 1] = one_class_error_diff[y == 1]
                error_diff = torch.tensor(error_diff)

                self.temperature = temperature
                weight = torch.exp(-error_diff / self.temperature)
                self.normalized_weight = weight / torch.sum(weight) * self.x.shape[0]

                df = pd.DataFrame(self.normalized_weight.numpy(), columns=["weight"])
                df = df.round({
                    "weight": 4
                })
                df.to_csv(f"{weight_store_path}/weight.csv")
            
            elif weight_criterion == "single_timestep_common_error":
                unweight_error = torch.load(f"./weights_discrete/{dataname}/unweight_error_2.pt").cpu()
                sigma = torch.load(f"./weights_discrete/{dataname}/sigma_2.pt").cpu()
                
                error = unweight_error[:, args.single_sigma_index, :, :]
                mean_error = torch.mean(error, dim=(1, 2))

                if args.error_reflection == "linear":
                    weight = mean_error
                elif args.error_reflection == "softmax":
                    self.temperature = temperature
                    weight = torch.exp(mean_error / self.temperature)
                self.normalized_weight = weight / torch.sum(weight) * self.x.shape[0]

                df = pd.DataFrame(self.normalized_weight.numpy(), columns=["weight"])
                df.to_csv(f"{weight_store_path}/weight.csv")

            elif weight_criterion == "single_timestep_error_diff":
                unweight_zero_model_error = torch.load(f"./weights_discrete/{dataname}/unweight_error_0.pt").cpu()
                unweight_one_model_error = torch.load(f"./weights_discrete/{dataname}/unweight_error_1.pt").cpu()
                
                zero_model_error = unweight_zero_model_error[:, args.single_sigma_index, :, :]
                zero_model_mean_error = torch.mean(zero_model_error, dim=(1, 2))
                one_model_error = unweight_one_model_error[:, args.single_sigma_index, :, :]
                one_model_mean_error = torch.mean(one_model_error, dim=(1, 2))

                zero_class_error_diff = one_model_mean_error - zero_model_mean_error
                one_class_error_diff = zero_model_mean_error - one_model_mean_error
                error_diff = np.zeros(self.y.shape[0])
                error_diff[y == 0] = zero_class_error_diff[y == 0]
                error_diff[y == 1] = one_class_error_diff[y == 1]
                error_diff = torch.tensor(error_diff)

                self.temperature = temperature
                weight = torch.exp(-error_diff / self.temperature)
                self.normalized_weight = weight / torch.sum(weight) * self.x.shape[0]

                df = pd.DataFrame(self.normalized_weight.numpy(), columns=["weight"])
                df = df.round({
                    "weight": 4
                })
                df.to_csv(f"{weight_store_path}/weight.csv")

            elif weight_criterion == "several_timestep_error_diff":
                # Original Experiment run with weight_discrete, num_timesteps=10, K=32
                unweight_zero_model_error = torch.load(f"./weights_discrete_10_{args.K}/{dataname}/unweight_error_0.pt").cpu()
                unweight_one_model_error = torch.load(f"./weights_discrete_10_{args.K}/{dataname}/unweight_error_1.pt").cpu()
                
                zero_model_sigma = torch.load(f"./weights_discrete_10_{args.K}/{dataname}/sigma_0.pt").cpu()
                one_model_sigma = torch.load(f"./weights_discrete_10_{args.K}/{dataname}/sigma_1.pt").cpu()

                if args.selected_several_sigma_indices != None:
                    print(f"use selected several timesteps: {args.selected_several_sigma_indices}")
                    unweight_zero_model_error = unweight_zero_model_error[:, args.selected_several_sigma_indices, :, :]
                    unweight_one_model_error = unweight_one_model_error[:, args.selected_several_sigma_indices, :]
                    zero_model_sigma = zero_model_sigma[:, args.selected_several_sigma_indices, :]
                    one_model_sigma = one_model_sigma[:, args.selected_several_sigma_indices, :]

                if args.timestep_weight_criterion == "down":
                    zero_model_time_weight = 1 / zero_model_sigma ** 2
                    one_model_time_weight = 1 / one_model_sigma ** 2
                elif args.timestep_weight_criterion == "avg":
                    zero_model_time_weight = torch.ones_like(zero_model_sigma)
                    one_model_time_weight = torch.ones_like(one_model_sigma)
                elif args.timestep_weight_criterion == "EDM":
                    sigma_data = 0.5
                    zero_model_time_weight = (zero_model_sigma ** 2 + sigma_data ** 2) / (zero_model_sigma * sigma_data) ** 2
                    one_model_time_weight = (one_model_sigma ** 2 + sigma_data ** 2) / (one_model_sigma * sigma_data) ** 2

                zero_model_time_weight = zero_model_time_weight.unsqueeze(-1)
                one_model_time_weight = one_model_time_weight.unsqueeze(-1)

                # print("model time weight", zero_model_time_weight.shape, one_model_time_weight.shape)
                # print("unweight model error", unweight_zero_model_error.shape, unweight_one_model_error.shape)

                zero_model_error = unweight_zero_model_error * zero_model_time_weight
                zero_model_mean_error = torch.mean(zero_model_error, dim=(1, 2, 3))
                one_model_error = unweight_one_model_error * one_model_time_weight
                one_model_mean_error = torch.mean(one_model_error, dim=(1, 2, 3))

                zero_class_error_diff = one_model_mean_error - zero_model_mean_error
                one_class_error_diff = zero_model_mean_error - one_model_mean_error
                error_diff = np.zeros(self.y.shape[0])
                error_diff[y == 0] = zero_class_error_diff[y == 0]
                error_diff[y == 1] = one_class_error_diff[y == 1]
                error_diff = torch.tensor(error_diff)

                self.temperature = temperature
                weight = torch.exp(-error_diff / self.temperature)
                self.normalized_weight = weight / torch.sum(weight) * self.x.shape[0]

                df = pd.DataFrame(self.normalized_weight.numpy(), columns=["weight"])
                df = df.round({
                    "weight": 4
                })
                df.to_csv(f"{weight_store_path}/weight.csv")

            else:
                assert False, f"No such weight criterion: {weight_criterion}"

        self.return_index = return_index
        self.attr = returned_attr
        if returned_attr is not None:
            raw_df = pd.read_csv(f"/data/my_stored_dataset/{dataname}/test.csv")
            attr_indicator = raw_df[returned_attr].values
            self.attr_indicator = self.preprocess_attributes(attr_indicator)


    def preprocess_attributes(self, attr_indicator):
        # numerical_attr_indicator = np.zeros_like(attr_indicator)
        if self.dataname == "adult":
            if self.attr == "sex":
                attr_indicator[attr_indicator == " Male"] = 0
                attr_indicator[attr_indicator == " Female"] = 1
            elif self.attr == "marital.status":
                attr_indicator[(attr_indicator == " Never-married") | (attr_indicator == " Divorced")] = 0
                attr_indicator[attr_indicator != 0] = 1
            elif self.attr == "race":
                attr_indicator[attr_indicator == " White"] = 0
                attr_indicator[attr_indicator != 0] = 1

        elif self.dataname == "shoppers":
            if self.attr == "Weekend":
                attr_indicator[attr_indicator == False] = 0
                attr_indicator[attr_indicator == True] = 1
            elif self.attr == "OperatingSystems":
                attr_indicator[attr_indicator <= 4] = 0
                attr_indicator[attr_indicator >= 5] = 1
            elif self.attr == "Browser":
                attr_indicator[attr_indicator <= 7] = 0
                attr_indicator[attr_indicator > 7] = 1
            elif self.attr == "VisitorType":
                attr_indicator[attr_indicator == "New_Visitor"] = 0
                attr_indicator[attr_indicator != 0] = 1
            elif self.attr == "TrafficType":
                attr_indicator[attr_indicator <= 2] = 0
                attr_indicator[attr_indicator > 2] = 1

        elif self.dataname == "default":
            if self.attr == "SEX":
                attr_indicator[attr_indicator == 1] = 0
                attr_indicator[attr_indicator == 2] = 1
            elif self.attr == "MARRIAGE":
                attr_indicator[attr_indicator == 1] = 1
                attr_indicator[attr_indicator != 1] = 0
            elif self.attr == "AGE":
                attr_indicator[attr_indicator <= 35] = 0
                attr_indicator[attr_indicator > 35] = 1
            elif self.attr == "LIMIT_BAL":
                attr_indicator[attr_indicator <= 160000] = 0
                attr_indicator[attr_indicator > 160000] = 1
                
        elif self.dataname == "bank":
            if self.attr == "marital":
                attr_indicator[attr_indicator == 'married'] = 1
                attr_indicator[attr_indicator != 1] = 0
            elif self.attr == "housing":
                attr_indicator[attr_indicator == 'no'] = 0
                attr_indicator[attr_indicator == 'yes'] = 1
            elif self.attr == "age":
                attr_indicator[attr_indicator <= 40] = 0
                attr_indicator[attr_indicator > 40] = 1
            elif self.attr == "duration":
                attr_indicator[attr_indicator <= 200] = 0
                attr_indicator[attr_indicator > 200] = 1

        elif self.dataname[:4] == "taxi":
            if self.attr == "month":
                if self.dataname[5:8] == "nyc":
                    threshold = 3
                else:
                    threshold = 7
                attr_indicator[attr_indicator <= threshold] = 0 
                attr_indicator[attr_indicator > threshold] = 1
            elif self.attr == "weekday":
                attr_indicator[attr_indicator <= 3] = 0
                attr_indicator[attr_indicator > 3] = 1
            elif self.attr == "hour":
                # EO itself is too low
                attr_indicator[attr_indicator <= 6] = 0
                attr_indicator[attr_indicator > 6] = 1
            elif self.attr == "direction":
                attr_indicator[attr_indicator <= 2] = 0
                attr_indicator[attr_indicator > 2] = 1

        elif self.dataname[:10] == "ACS_income":
            if self.attr == "SEX":
                attr_indicator[attr_indicator == 2] = 0
                attr_indicator[attr_indicator == 1] = 1
            elif self.attr == "race":
                attr_indicator[attr_indicator != 0] = 1 # white 0, others 1
            elif self.attr == "relp":
                attr_indicator[attr_indicator != 0] = 1 # reference 0
                

        return attr_indicator


    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        if self.attr is None: # training 
            if self.use_weight:
                return self.x[index], self.y[index], self.normalized_weight[index]
            else:
                if self.return_index:
                    return self.x[index], self.y[index], index # used by LfF
                else:
                    return self.x[index], self.y[index]
        else: # testing
            if self.return_index:
                return self.x[index], self.y[index], self.attr_indicator[index], index
            else:
                return self.x[index], self.y[index], self.attr_indicator[index]


    

