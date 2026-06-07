from ..quant_layers.matmul import PTQSLBatchingQuantMatMul
import torch
import torch.nn as nn
import torch.nn.functional as F

class MinMaxQuantLinear(nn.Linear):
    def __init__(self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode = "raw",
        w_bit = 8,
        a_bit = 8,
        bias_bit = None,
        bias_correction=False):
        super().__init__(in_features,out_features,bias)
        self.n_calibration_step=2
        self.mode = mode
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.bias_bit=bias_bit
        assert bias_bit is None,"No support bias bit now"
        self.w_interval=None
        self.a_interval=None
        self.raw_input=None
        self.raw_out=None
        self.metric=None
        self.next_nodes=[]
        self.w_qmax=2**(self.w_bit-1)
        self.a_qmax=2**(self.a_bit-1)
        self.bias_correction = bias_correction

    def forward(self, x):
        if self.mode=='raw':
            out=F.linear(x, self.weight, self.bias)
        elif self.mode=="quant_forward":
            out=self.quant_forward(x)
        elif self.mode=="calibration_step1":
            out=self.calibration_step1(x)
        elif self.mode=="calibration_step2":
            out=self.calibration_step2(x)
        else:
            raise NotImplementedError
        return out
    
    def quant_weight_bias(self):
        w=(self.weight/self.w_interval).round_().clamp_(-self.w_qmax,self.w_qmax-1)
        w_sim=w.mul_(self.w_interval)
        if self.bias is not None:
            return w_sim,self.bias
            # bias=(self.bias/self.bias_interval).round_().clamp_(-self.bias_qmax,self.bias_qmax-1)
            # bias_sim=bias*self.bias_interval
            # return w_sim,bias_sim
        else:
            return w_sim,None
    
    def quant_input(self, x):
        x_sim=(x/self.a_interval).round_().clamp_(-self.a_qmax,self.a_qmax-1)
        x_sim.mul_(self.a_interval)
        return x_sim
    
    def quant_forward(self,x):
        assert self.calibrated is not None,f"You should run calibrate_forward before run quant_forward for {self}"
        w_sim,bias_sim=self.quant_weight_bias()
        x_sim=self.quant_input(x)
        out=F.linear(x_sim, w_sim, bias_sim)
        return out
    
    def _bias_correction_quant_forward(self, x):
        if self.bias_correction and self.bias != None:
            w_sim = self.quant_weight_bias()[0]
            x_sim = self.quant_input(x)
            eps = F.linear(x_sim, w_sim-self.weight.data, None)
            eps = torch.mean(eps, dim=(list(range(len(eps.shape)-1))), keepdim=False)
            self.bias -= eps
            self.bias_correction = False
        return self.quant_forward(x)

    def calibration_step1(self,x):
        # step1: collection the FP32 values
        out=F.linear(x, self.weight, self.bias)
        self.raw_input=x.cpu().detach()
        self.raw_out=out.cpu().detach()
        return out
    
    def calibration_step2(self,x):
        # step2: search for the best S^w and S^o of each layer
        self.w_interval=(self.weight.data.abs().max()/(self.w_qmax-0.5)).detach()
        self.a_interval=(x.abs().max()/(self.a_qmax-0.5)).detach()
        self.calibrated=True
        out=self._bias_correction_quant_forward(x)
        return out

class PTQSLQuantLinear(MinMaxQuantLinear):
    """
    PTQSL on linear modules.
    """
    def __init__(self, 
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode = "raw",
        w_bit = 8,
        a_bit = 8,
        bias_bit = None,
        bias_correction = False,
        metric="L2_norm", search_round=1, eq_alpha=0, eq_beta=1, eq_n=100, parallel_eq_n=10, n_H=1, n_V=1, n_a=1, init_layerwise=False):
        super().__init__(in_features, out_features, bias=bias, mode=mode, w_bit=w_bit, a_bit=a_bit, bias_bit=bias_bit, bias_correction=bias_correction)
        self.metric = metric
        self.search_round = search_round
        self.eq_alpha = eq_alpha
        self.eq_beta = eq_beta
        self.eq_n = eq_n
        self.n_H = n_H
        self.n_V = n_V
        self.n_a = n_a
        self.crb_rows = out_features // n_V
        self.crb_cols = in_features // n_H # ignore remnent != 0 situations
        self.crb_acts = in_features // n_a
        self.parallel_eq_n = parallel_eq_n
        self.init_layerwise = init_layerwise
        self.raw_grad = None

    def _get_similarity(self, tensor_raw, tensor_sim, metric=None):
        """
        tensor_raw: *, features
        tensor_sim: *, features
        similarity: *
        It's your job to calculate mean on * dims!
        """
        if metric == "cosine":
            similarity = F.cosine_similarity(tensor_raw, tensor_sim, dim=-1)
        elif metric == "pearson":
            similarity = F.cosine_similarity(tensor_raw-torch.mean(tensor_raw,dim=-1,keepdim=True), tensor_sim-torch.mean(tensor_sim,dim=-1,keepdim=True), dim=-1)
        else:
            if metric == "L1_norm":
                similarity = -torch.abs(tensor_raw - tensor_sim)
            elif metric == "L2_norm":
                similarity = -(tensor_raw - tensor_sim) ** 2
            elif metric == "linear_weighted_L2_norm":
                similarity = -tensor_raw.abs() * (tensor_raw - tensor_sim) ** 2
            elif metric == "square_weighted_L2_norm":
                similarity = -(tensor_raw * (tensor_raw - tensor_sim)) ** 2
            elif metric == "hessian":
                raw_grad = self.raw_grad.reshape_as(tensor_raw)
                similarity = -(raw_grad * (tensor_raw - tensor_sim)) ** 2
            else:
                raise NotImplementedError(f"metric {metric} not implemented!")
            similarity = torch.mean(similarity, dim=-1)
        return similarity
    
    def quant_weight_bias(self):
        # self.w_interval shape: n_V, 1, n_H, 1
        w=(self.weight.view(self.n_V,self.crb_rows,self.n_H,self.crb_cols)/self.w_interval).round_().clamp_(-self.w_qmax,self.w_qmax-1)
        w_sim=w.mul_(self.w_interval).view(self.out_features,self.in_features)
        if self.bias is not None:
            return w_sim,self.bias
            # bias=(self.bias/self.bias_interval).round_().clamp_(-self.bias_qmax,self.bias_qmax-1)
            # bias_sim=bias*self.bias_interval
            # return w_sim,bias_sim
        else:
            return w_sim,None
    
    def quant_input(self, x):
        # self.a_interval shape: n_a,1
        x_sim=torch.cat(torch.chunk(x.unsqueeze(-2), chunks=self.n_a, dim=-1), dim=-2)
        x_sim=(x_sim.div_(self.a_interval)).round_().clamp_(-self.a_qmax,self.a_qmax-1)
        x_sim = x_sim.mul_(self.a_interval).reshape_as(x)
        return x_sim

    def _search_best_w_interval(self, x, weight_interval_candidates, raw_out_expanded_chunked):
        """
        Modularization of searching best weight intervals
        """
        tmp_w_interval = self.w_interval.unsqueeze(0) # shape: 1,n_V,1,n_H,1
        for h in range(self.n_H):
            similarities = []
            for p_st in range(0,self.eq_n,self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                cur_w_interval = tmp_w_interval.repeat(p_ed-p_st,1,1,1,1)
                cur_w_interval[:,:,:,h:h+1,:] = weight_interval_candidates[p_st:p_ed,:,:,h:h+1,:]
                # quantize weight and bias 
                w_sim = self.weight.view(self.n_V,self.crb_rows,self.n_H,self.crb_cols).unsqueeze(0) # shape: 1,n_V,crb_rows,n_H,crb_cols
                w_sim = (w_sim/cur_w_interval).round_().clamp_(-self.w_qmax,self.w_qmax-1).mul_(cur_w_interval) # shape: parallel_eq_n,n_V,crb_rows,n_H,crb_cols
                w_sim = w_sim.view(-1,self.in_features) # shape: parallel_eq_n*oc,ic
                bias_sim = self.bias.repeat(p_ed-p_st) if self.bias is not None else None
                # quantize input
                x_sim = self.quant_input(x)
                # calculate similarity and store them
                out_sim = F.linear(x_sim, w_sim, bias_sim) # shape: B,*,parallel_eq_n*oc
                out_sim = torch.cat(torch.chunk(out_sim.unsqueeze(-2), chunks=p_ed-p_st, dim=-1), dim=-2) # shape: B,*,parallel_eq_n,oc
                out_sim = torch.cat(torch.chunk(out_sim.unsqueeze(-2), chunks=self.n_V, dim=-1), dim=-2) # shape: B,*,parallel_eq_n,n_V,crb_rows
                similarity = self._get_similarity(raw_out_expanded_chunked, out_sim, self.metric) # shape: B,*,parallel_eq_n,n_V
                similarity = torch.mean(similarity, dim=list(range(len(similarity.shape)-2))) # shape: parallel_eq_n, n_V
                similarities.append(similarity)
            # store best weight interval of h into tmp_w_interval
            similarities = torch.cat(similarities, dim=0) # shape: eq_n, n_V
            h_best_index = similarities.argmax(dim=0).reshape(1,-1,1,1,1) # shape: 1,n_V,1,1,1
            tmp_w_interval[:,:,:,h:h+1,:] = torch.gather(weight_interval_candidates[:,:,:,h:h+1,:],dim=0,index=h_best_index)
        self.w_interval = tmp_w_interval.squeeze(dim=0)
    
    def _search_best_a_interval(self, x, input_interval_candidates, raw_out_expanded):
        tmp_a_interval = self.a_interval.unsqueeze(-1) # shape: n_a,1,1
        for a in range(self.n_a):
            similarities = []
            for p_st in range(0,self.eq_n,self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                cur_a_interval = tmp_a_interval.repeat(1,1,p_ed-p_st) # shape: n_a,1,parallel_eq_n
                cur_a_interval[a:a+1,:,:] = input_interval_candidates[a:a+1,:,p_st:p_ed]
                # quantize weight and bias 
                w_sim, bias_sim = self.quant_weight_bias()
                # quantize input
                x_sim=torch.cat(torch.chunk(x.unsqueeze(-2), chunks=self.n_a, dim=-1), dim=-2).unsqueeze(-1)
                x_sim=(x_sim/(cur_a_interval)).round_().clamp_(-self.a_qmax,self.a_qmax-1)*(cur_a_interval) # shape: B,*,n_a,crb_acts,parallel_eq_n
                x_sim = x_sim.permute(*list(range(len(x_sim.shape)-3)),-1,-3,-2).reshape(*x.shape[:-1],p_ed-p_st,x.shape[-1]) # shape: B,*,parallel_eq_n,ic
                # calculate similarity and store them
                out_sim = F.linear(x_sim, w_sim, bias_sim) # shape: B,*,parallel_eq_n,oc
                similarity = self._get_similarity(raw_out_expanded, out_sim, self.metric) # shape: B,*,parallel_eq_n
                similarity = torch.mean(similarity, dim=list(range(len(similarity.shape)-1))) # shape: parallel_eq_n
                similarities.append(similarity)
            # store best input interval and store in tmp_a_interval
            similarities = torch.cat(similarities, dim=0) # shape: eq_n
            a_best_index = similarities.argmax(dim=0, keepdim=True).reshape(1,1,-1)
            tmp_a_interval[a:a+1,:,:] = torch.gather(input_interval_candidates[a:a+1,:,:],dim=2,index=a_best_index)
        self.a_interval = tmp_a_interval.squeeze(-1)

    def _initialize_intervals(self, x):
        if self.init_layerwise:
            self.w_interval=((self.weight.abs().max())/(self.w_qmax-0.5)).view(1,1,1,1).repeat(self.n_V,1,self.n_H,1)
            self.a_interval=(x.abs().max()/(self.a_qmax-0.5)).detach().view(1,1).repeat(self.n_a,1)
        else:
            self.w_interval=(self.weight.view(self.n_V,self.crb_rows,self.n_H,self.crb_cols).abs().amax([1,3],keepdim=True)/(self.w_qmax-0.5))
            self.a_interval=((x.view(*x.shape[:-1],self.n_a,self.crb_acts).abs().amax(list(range(len(x.shape)-1))+[-1],keepdim=False))/(self.a_qmax-0.5)).unsqueeze(-1)

    def calibration_step2(self,x):
        # initialize intervals with minmax intervals
        self._initialize_intervals(x)

        # put raw outs on GPU
        raw_out_expanded = self.raw_out.to(x.device).unsqueeze(-2)  # shape: B,*,1,oc
        raw_out_expanded_chunked = torch.cat(torch.chunk(raw_out_expanded.unsqueeze(-2), chunks=self.n_V, dim=-1), dim=-2) # shape: B,*,1,n_V,crb_rows

        # put raw grad on GPU
        self.raw_grad = self.raw_grad.to(x.device) if self.raw_grad != None else None

        # prepare weight intervals and similarities
        weight_interval_candidates = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).cuda().view(-1,1,1,1,1) * self.w_interval.unsqueeze(0) # shape: eq_n,n_V,1,n_H,1
        input_interval_candidates =  torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).cuda().view(1,1,-1) * self.a_interval.unsqueeze(-1) # shape: n_a,1,eq_n
        for e in range(self.search_round):
            # search for best weight interval
            self._search_best_w_interval(x, weight_interval_candidates, raw_out_expanded_chunked)
            # search for best input interval
            self._search_best_a_interval(x, input_interval_candidates, raw_out_expanded)

        self.raw_grad = self.raw_grad.to("cpu") if self.raw_grad != None else None

        self.calibrated = True
        out=self._bias_correction_quant_forward(x)
        del self.raw_input, self.raw_out, self.raw_grad
        return out    

class PostGeluPTQSLQuantLinear(PTQSLQuantLinear):
    def __init__(self, 
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode = "raw",
        w_bit = 8,
        a_bit = 8,
        bias_bit = None,
        bias_correction = False,
        metric="L2_norm", search_round=1, eq_alpha=0, eq_beta=1, eq_n=100, parallel_eq_n=10, n_H=1, n_V=1, n_a=1, init_layerwise=False):
        super().__init__(in_features, out_features, bias=bias, mode=mode, w_bit=w_bit, a_bit=a_bit, bias_bit=bias_bit, bias_correction=bias_correction,
                         metric=metric, search_round=search_round, eq_alpha=eq_alpha, eq_beta=eq_beta, eq_n=eq_n, parallel_eq_n=parallel_eq_n, n_H=n_H, n_V=n_V, n_a=n_a, init_layerwise=init_layerwise)
    
    def quant_input(self, x):
        """
        self.a_interval = [a_interval_pos, a_interval_neg]
        """
        # self.a_interval[0] shape: n_a,1
        # self.a_interval[1] shape: 1
        x_=torch.cat(torch.chunk(x.unsqueeze(-2), chunks=self.n_a, dim=-1), dim=-2)
        x_pos=(x_/(self.a_interval[0])).round_().clamp_(0,self.a_qmax-1).mul_(self.a_interval[0])
        x_neg=(x_/(self.a_interval[1])).round_().clamp_(-self.a_qmax,0).mul_(self.a_interval[1])
        return (x_pos + x_neg).reshape_as(x)

    def _search_best_a_interval(self, x, input_interval_candidates, raw_out_expanded):
        tmp_a_interval = self.a_interval[0].unsqueeze(-1) # shape: n_a,1,1
        for a in range(self.n_a):
            similarities = []
            for p_st in range(0,self.eq_n,self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                cur_a_interval = tmp_a_interval.repeat(1,1,p_ed-p_st) # shape: n_a,1,parallel_eq_n
                cur_a_interval[a:a+1,:,:] = input_interval_candidates[a:a+1,:,p_st:p_ed]
                # quantize weight and bias 
                w_sim, bias_sim = self.quant_weight_bias()
                # quantize input
                x_sim=torch.cat(torch.chunk(x.unsqueeze(-2), chunks=self.n_a, dim=-1), dim=-2).unsqueeze(-1)
                x_pos=(x_sim/(cur_a_interval)).round_().clamp_(0,self.a_qmax-1)*(cur_a_interval) # shape: B,*,n_a,crb_acts,parallel_eq_n
                x_neg=(x_sim/(self.a_interval[1])).round_().clamp_(-self.a_qmax,0)*(self.a_interval[1]) # shape: B,*,n_a,crb_acts,1
                x_sim = (x_pos + x_neg).permute(*list(range(len(x_sim.shape)-3)),-1,-3,-2).reshape(*x.shape[:-1],p_ed-p_st,x.shape[-1]) # shape: B,*,parallel_eq_n,ic
                # calculate similarity and store them
                out_sim = F.linear(x_sim, w_sim, bias_sim) # shape: B,*,parallel_eq_n,oc
                similarity = self._get_similarity(raw_out_expanded, out_sim, self.metric) # shape: B,*,parallel_eq_n
                similarity = torch.mean(similarity, dim=list(range(len(similarity.shape)-1))) # shape: parallel_eq_n
                similarities.append(similarity)
            # store best input interval and store in tmp_a_interval
            similarities = torch.cat(similarities, dim=0) # shape: eq_n
            a_best_index = similarities.argmax(dim=0, keepdim=True).reshape(1,1,-1)
            tmp_a_interval[a:a+1,:,:] = torch.gather(input_interval_candidates[a:a+1,:,:],dim=2,index=a_best_index)
        self.a_interval[0] = tmp_a_interval.squeeze(-1)

    def _initialize_intervals(self, x):
        if self.init_layerwise:
            self.w_interval=((self.weight.abs().max())/(self.w_qmax-0.5)).view(1,1,1,1).repeat(self.n_V,1,self.n_H,1)
            self.a_interval=[(x.max()/(self.a_qmax-0.5)).detach().view(1,1).repeat(self.n_a,1)]
        else:
            self.w_interval=(self.weight.view(self.n_V,self.crb_rows,self.n_H,self.crb_cols).abs().amax([1,3],keepdim=True)/(self.w_qmax-0.5))
            self.a_interval=[((x.view(*x.shape[:-1],self.n_a,self.crb_acts).amax(list(range(len(x.shape)-1))+[-1],keepdim=False))/(self.a_qmax-0.5)).unsqueeze(-1)]
        self.a_interval.append(0.16997124254703522/self.a_qmax)

    def calibration_step2(self,x):
        # initialize intervals with minmax intervals
        self._initialize_intervals(x)

        # put raw outs on GPU
        raw_out_expanded = self.raw_out.to(x.device).unsqueeze(-2)  # shape: B,*,1,oc
        raw_out_expanded_chunked = torch.cat(torch.chunk(raw_out_expanded.unsqueeze(-2), chunks=self.n_V, dim=-1), dim=-2) # shape: B,*,1,n_V,crb_rows

        # put raw grad on GPU
        self.raw_grad = self.raw_grad.to(x.device) if self.raw_grad != None else None

        # prepare weight intervals and similarities
        weight_interval_candidates = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).cuda().view(-1,1,1,1,1) * self.w_interval.unsqueeze(0) # shape: eq_n,n_V,1,n_H,1
        input_interval_candidates =  torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).cuda().view(1,1,-1) * self.a_interval[0].unsqueeze(-1) # shape: n_a,1,eq_n
        for e in range(self.search_round):
            # search for best weight interval
            self._search_best_w_interval(x, weight_interval_candidates, raw_out_expanded_chunked)
            # search for best input interval
            self._search_best_a_interval(x, input_interval_candidates, raw_out_expanded)

        self.raw_grad = self.raw_grad.to("cpu") if self.raw_grad != None else None

        self.calibrated = True
        out=self._bias_correction_quant_forward(x)
        del self.raw_input, self.raw_out, self.raw_grad
        return out    

class PTQSLBatchingQuantLinear(PTQSLQuantLinear):
    def __init__(self, 
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode = "raw",
        w_bit = 8,
        a_bit = 8,
        bias_bit = None,
        bias_correction = False,
        metric="L2_norm", search_round=1, eq_alpha=0, eq_beta=1, eq_n=100, parallel_eq_n=10, n_H=1, n_V=1, n_a=1, init_layerwise=False):
        super().__init__(in_features, out_features, bias=bias, mode=mode, w_bit=w_bit, a_bit=a_bit, bias_bit=bias_bit, bias_correction=bias_correction, metric=metric, search_round=search_round, eq_alpha=eq_alpha, eq_beta=eq_beta, eq_n=eq_n, parallel_eq_n=parallel_eq_n, n_H=n_H, n_V=n_V, n_a=n_a, init_layerwise=init_layerwise)
        self.calib_size = None
        self.calib_batch_size = None
        self.calib_need_batching = False

    def _initialize_calib_parameters(self):
        """ 
        set parameters for feeding calibration data
        """
        self.calib_size = int(self.raw_input.shape[0])
        self.calib_batch_size = int(self.raw_input.shape[0])
        while True:
            numel = (2*(self.raw_input.numel()+self.raw_out.numel())/self.calib_size*self.calib_batch_size) # number of parameters on GPU
            self.parallel_eq_n = int((3*1024*1024*1024/4)//numel)
            if self.parallel_eq_n <= 1:
                self.calib_need_batching = True
                self.calib_batch_size //= 2
            else:
                break
    
    def _initialize_intervals(self):
        # weight intervals 
        if self.init_layerwise:
            self.w_interval=((self.weight.abs().max())/(self.w_qmax-0.5)).view(1,1,1,1).repeat(self.n_V,1,self.n_H,1)
        else:
            self.w_interval=(self.weight.view(self.n_V,self.crb_rows,self.n_H,self.crb_cols).abs().amax([1,3],keepdim=True)/(self.w_qmax-0.5))

        # activation intervals
        tmp_a_intervals = []
        for b_st in range(0,self.calib_size,self.calib_batch_size):
            b_ed = min(self.calib_size, b_st+self.calib_batch_size)
            x_ = self.raw_input[b_st:b_ed].cuda()
            if self.init_layerwise:
                a_interval_=(x_.abs().max()/(self.a_qmax-0.5)).detach().view(1,1).repeat(self.n_a,1)
            else:
                a_interval_=((x_.view(*x_.shape[:-1],self.n_a,self.crb_acts).abs().amax(list(range(len(x_.shape)-1))+[-1],keepdim=False))/(self.a_qmax-0.5)).unsqueeze(-1)
            tmp_a_intervals.append(a_interval_)
        self.a_interval = torch.cat(tmp_a_intervals, dim=1).amax(dim=1, keepdim=True)

    def _get_similarity(self, tensor_raw, tensor_sim, metric=None, raw_grad=None):
        """
        tensor_raw: *, features
        tensor_sim: *, features
        similarity: *
        It's your job to calculate mean on * dims!
        """
        if metric == "cosine":
            similarity = F.cosine_similarity(tensor_raw, tensor_sim, dim=-1)
        else:
            if metric == "L1_norm":
                similarity = -torch.abs(tensor_raw - tensor_sim)
            elif metric == "L2_norm":
                similarity = -(tensor_raw - tensor_sim) ** 2
            elif metric == "linear_weighted_L2_norm":
                similarity = -tensor_raw.abs() * (tensor_raw - tensor_sim) ** 2
            elif metric == "square_weighted_L2_norm":
                similarity = -(tensor_raw * (tensor_raw - tensor_sim)) ** 2
            elif metric == "hessian":
                assert raw_grad != None, f"raw_grad is None in _get_similarity!"
                raw_grad = raw_grad.reshape_as(tensor_raw)
                similarity = -(raw_grad * (tensor_raw - tensor_sim)) ** 2
            else:
                raise NotImplementedError(f"metric {metric} not implemented!")
            similarity = torch.mean(similarity, dim=-1)
        return similarity

    def _get_pearson_w(self, tensor_raw, tensor_sim):
        """
        Quick implementation of similarity-aware linear quantization
        tensor_sim: b,*,parallel_eq_n,n_V,crb_rows
        tensor_raw: b,*,1,n_V,crb_rows
        """
        b, parallel_eq_n, n_V = tensor_sim.shape[0],tensor_sim.shape[-3],tensor_sim.shape[-2]
        tensor_sim = tensor_sim.transpose(-1,-3).contiguous_().view(b,-1,n_V,parallel_eq_n)
        tensor_raw = tensor_raw.transpose(-1,-3).view(b,-1,n_V,1)
        tensor_sim_mean = tensor_sim.mean(dim=[0,1],keepdim=True)
        tensor_raw_mean = tensor_raw.mean(dim=[0,1],keepdim=True)
        similarity = torch.cosine_similarity(tensor_raw-tensor_raw_mean, tensor_sim-tensor_sim_mean, dim=1) # shape: b,n_V,parallel_eq_n
        similarity = similarity.permute(0,2,1).contiguous_()
        return similarity
    
    def _get_pearson_a(self, tensor_raw, tensor_sim):
        """
        Quick implementation of similarity-aware linear quantization
        tensor_sim: b,*,parallel_eq_n,oc
        tensor_raw: b,*,1,oc
        """
        b, parallel_eq_n = tensor_sim.shape[0],tensor_sim.shape[-2]
        tensor_sim = tensor_sim.transpose(-1,-2).contiguous_().view(b,-1,parallel_eq_n)
        tensor_raw = tensor_raw.transpose(-1,-2).view(b,-1,1)
        tensor_sim_mean = tensor_sim.mean(dim=[0,1],keepdim=True)
        tensor_raw_mean = tensor_raw.mean(dim=[0,1],keepdim=True)
        similarity = torch.cosine_similarity(tensor_raw-tensor_raw_mean, tensor_sim-tensor_sim_mean, dim=1) # shape: b,parallel_eq_n
        return similarity

    def _search_best_w_interval(self, weight_interval_candidates):
        tmp_w_interval = self.w_interval.unsqueeze(0) # shape: 1,n_V,1,n_H,1
        for h in range(self.n_H):
            batch_similarities = [] # similarities, need to concatenate and calculate sum (equivalent to mean with argmax)
            for b_st in range(0, self.calib_size, self.calib_batch_size):
                b_ed = min(self.calib_size, b_st + self.calib_batch_size)
                x = self.raw_input[b_st:b_ed].cuda()
                raw_out_expanded = self.raw_out[b_st:b_ed].cuda().unsqueeze(-2) # shape: b,*,1,oc
                raw_out_expanded = torch.cat(torch.chunk(raw_out_expanded.unsqueeze(-2), chunks=self.n_V, dim=-1), dim=-2) # shape: b,*,1,n_V,crb_rows
                raw_grad = self.raw_grad[b_st:b_ed].cuda() # will be reshaped later
                similarities = []
                for p_st in range(0,self.eq_n,self.parallel_eq_n):
                    p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                    cur_w_interval = tmp_w_interval.repeat(p_ed-p_st,1,1,1,1)
                    cur_w_interval[:,:,:,h:h+1,:] = weight_interval_candidates[p_st:p_ed,:,:,h:h+1,:]
                    # quantize weight and bias 
                    w_sim = self.weight.view(self.n_V,self.crb_rows,self.n_H,self.crb_cols).unsqueeze(0) # shape: 1,n_V,crb_rows,n_H,crb_cols
                    w_sim = (w_sim/cur_w_interval).round_().clamp_(-self.w_qmax,self.w_qmax-1).mul_(cur_w_interval) # shape: parallel_eq_n,n_V,crb_rows,n_H,crb_cols
                    w_sim = w_sim.view(-1,self.in_features) # shape: parallel_eq_n*oc,ic
                    bias_sim = self.bias.repeat(p_ed-p_st) if self.bias is not None else None
                    # quantize input
                    x_sim = self.quant_input(x)
                    # calculate similarity and store them
                    out_sim = F.linear(x_sim, w_sim, bias_sim) # shape: b,*,parallel_eq_n*oc
                    out_sim = torch.cat(torch.chunk(out_sim.unsqueeze(-2), chunks=p_ed-p_st, dim=-1), dim=-2) # shape: b,*,parallel_eq_n,oc
                    out_sim = torch.cat(torch.chunk(out_sim.unsqueeze(-2), chunks=self.n_V, dim=-1), dim=-2) # shape: b,*,parallel_eq_n,n_V,crb_rows
                    if self.metric != "pearson":
                        similarity = self._get_similarity(raw_out_expanded, out_sim, self.metric, raw_grad) # shape: b,*,parallel_eq_n,n_V
                        if len(similarity.shape) > 3:
                            similarity = torch.mean(similarity, dim=list(range(1,len(similarity.shape)-2))) # shape: b, parallel_eq_n, n_V
                    else:
                        similarity = self._get_pearson_w(raw_out_expanded, out_sim)
                    similarity = similarity.sum(dim=0, keepdim=True) # shape: 1, parallel_eq_n, n_V
                    similarities.append(similarity)
                # store best weight interval of h into tmp_w_interval
                similarities = torch.cat(similarities, dim=1) # shape: 1, eq_n, n_V
                batch_similarities.append(similarities)
            batch_similarities = torch.cat(batch_similarities, dim=0).sum(dim=0, keepdim=False) # shape: eq_n, n_V
            h_best_index = batch_similarities.argmax(dim=0).reshape(1,-1,1,1,1) # shape: 1,n_V,1,1,1
            tmp_w_interval[:,:,:,h:h+1,:] = torch.gather(weight_interval_candidates[:,:,:,h:h+1,:],dim=0,index=h_best_index)
        self.w_interval = tmp_w_interval.squeeze(dim=0)

    def _search_best_a_interval(self, input_interval_candidates):
        tmp_a_interval = self.a_interval.unsqueeze(-1) # shape: n_a,1,1
        for a in range(self.n_a):
            batch_similarities = [] # similarities, need to concatenate and calculate sum (equivalent to mean with argmax)
            for b_st in range(0, self.calib_size, self.calib_batch_size):
                b_ed = min(self.calib_size, b_st + self.calib_batch_size)
                x = self.raw_input[b_st:b_ed].cuda()
                raw_out_expanded = self.raw_out[b_st:b_ed].cuda().unsqueeze(-2) # shape: b,*,1,oc
                raw_grad = self.raw_grad[b_st:b_ed].cuda() # will be reshaped later
                similarities = []
                for p_st in range(0,self.eq_n,self.parallel_eq_n):
                    p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                    cur_a_interval = tmp_a_interval.repeat(1,1,p_ed-p_st) # shape: n_a,1,parallel_eq_n
                    cur_a_interval[a:a+1,:,:] = input_interval_candidates[a:a+1,:,p_st:p_ed]
                    # quantize weight and bias 
                    w_sim, bias_sim = self.quant_weight_bias()
                    # quantize input
                    x_sim=torch.cat(torch.chunk(x.unsqueeze(-2), chunks=self.n_a, dim=-1), dim=-2).unsqueeze(-1)
                    x_sim=(x_sim/(cur_a_interval)).round_().clamp_(-self.a_qmax,self.a_qmax-1)*(cur_a_interval) # shape: b,*,n_a,crb_acts,parallel_eq_n
                    x_sim = x_sim.permute(*list(range(len(x_sim.shape)-3)),-1,-3,-2).reshape(*x.shape[:-1],p_ed-p_st,x.shape[-1]) # shape: b,*,parallel_eq_n,ic
                    # calculate similarity and store them
                    out_sim = F.linear(x_sim, w_sim, bias_sim) # shape: b,*,parallel_eq_n,oc
                    if self.metric != "pearson":
                        similarity = self._get_similarity(raw_out_expanded, out_sim, self.metric, raw_grad) # shape: b,*,parallel_eq_n
                        if len(similarity.shape) > 2:
                            similarity = torch.mean(similarity, dim=list(range(1,len(similarity.shape)-1))) # shape: b, parallel_eq_n
                    else:
                        similarity = self._get_pearson_a(raw_out_expanded, out_sim)
                    similarity = torch.sum(similarity, dim=0, keepdim=True) # shape: 1, parallel_eq_n
                    similarities.append(similarity)
                # store best input interval and store in tmp_a_interval
                similarities = torch.cat(similarities, dim=1) # shape: 1, eq_n
                batch_similarities.append(similarities)
            batch_similarities = torch.cat(batch_similarities, dim=0).sum(dim=0, keepdim=False) # shape: eq_n
            a_best_index = batch_similarities.argmax(dim=0, keepdim=True).reshape(1,1,-1)
            tmp_a_interval[a:a+1,:,:] = torch.gather(input_interval_candidates[a:a+1,:,:],dim=2,index=a_best_index)
        self.a_interval = tmp_a_interval.squeeze(-1)


    def calibration_step2(self):
        """
        Only use cached raw inputs/outs/grads
        """
        self._initialize_calib_parameters()
        self._initialize_intervals()

        # prepare weight intervals and similarities
        weight_interval_candidates = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).cuda().view(-1,1,1,1,1) * self.w_interval.unsqueeze(0) # shape: eq_n,n_V,1,n_H,1
        input_interval_candidates =  torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).cuda().view(1,1,-1) * self.a_interval.unsqueeze(-1) # shape: n_a,1,eq_n
        for e in range(self.search_round):
            # search for best weight interval
            self._search_best_w_interval(weight_interval_candidates)
            # search for best input interval
            self._search_best_a_interval(input_interval_candidates)

        self.calibrated = True
        # self._bias_correction_quant_forward(self.raw_input.cuda()) # debugging
        del self.raw_input, self.raw_out, self.raw_grad
        return None

class PostGeluPTQSLBatchingQuantLinear(PTQSLBatchingQuantLinear):
    """ 
    An Agile implementation of PostGeluPTQSLBatchingQuantLinear
    use a_interval for positive activation quantization and a_neg_interval for negative activation quantization
    """
    def __init__(self, 
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode = "raw",
        w_bit = 8,
        a_bit = 8,
        bias_bit = None,
        bias_correction = False,
        metric="L2_norm", search_round=1, eq_alpha=0, eq_beta=1, eq_n=100, parallel_eq_n=10, n_H=1, n_V=1, n_a=1, init_layerwise=False):
        super().__init__(in_features, out_features, bias=bias, mode=mode, w_bit=w_bit, a_bit=a_bit, bias_bit=bias_bit, bias_correction=bias_correction,
                         metric=metric, search_round=search_round, eq_alpha=eq_alpha, eq_beta=eq_beta, eq_n=eq_n, parallel_eq_n=parallel_eq_n, n_H=n_H, n_V=n_V, n_a=n_a, init_layerwise=init_layerwise)
        self.a_neg_interval = 0.16997124254703522/self.a_qmax

    def _initialize_intervals(self):
        # weight intervals 
        if self.init_layerwise:
            self.w_interval=((self.weight.abs().max())/(self.w_qmax-0.5)).view(1,1,1,1).repeat(self.n_V,1,self.n_H,1)
        else:
            self.w_interval=(self.weight.view(self.n_V,self.crb_rows,self.n_H,self.crb_cols).abs().amax([1,3],keepdim=True)/(self.w_qmax-0.5))

        # activation intervals (for positive parts)
        if self.init_layerwise:
            tmp_a_intervals = []
            for b_st in range(0,self.calib_size,self.calib_batch_size):
                b_ed = min(self.calib_size, b_st+self.calib_batch_size)
                x_ = self.raw_input[b_st:b_ed].cuda()
                a_interval_=(x_.max()/(self.a_qmax-0.5)).detach().view(1,1).repeat(self.n_a,1)
                tmp_a_intervals.append(a_interval_)
            self.a_interval = torch.cat(tmp_a_intervals, dim=1).amax(dim=1, keepdim=True)
        else:
            tmp_a_intervals = []
            for b_st in range(0,self.calib_size,self.calib_batch_size):
                b_ed = min(self.calib_size, b_st+self.calib_batch_size)
                x_ = self.raw_input[b_st:b_ed].cuda()
                a_interval_=((x_.view(*x_.shape[:-1],self.n_a,self.crb_acts).amax(list(range(len(x_.shape)-1))+[-1],keepdim=False))/(self.a_qmax-0.5)).unsqueeze(-1)
                tmp_a_intervals.append(a_interval_)
            self.a_interval = torch.cat(tmp_a_intervals, dim=1).amax(dim=1, keepdim=True)

    def quant_input(self, x):
        # self.a_interval shape: n_a,1
        # self.a_neg_interval shape: 1
        x_=torch.cat(torch.chunk(x.unsqueeze(-2), chunks=self.n_a, dim=-1), dim=-2)
        x_pos=(x_/(self.a_interval)).round_().clamp_(0,self.a_qmax-1).mul_(self.a_interval)
        x_neg=(x_/(self.a_neg_interval)).round_().clamp_(-self.a_qmax,0).mul_(self.a_neg_interval)
        return (x_pos + x_neg).reshape_as(x)

    def _search_best_a_interval(self, input_interval_candidates):
        tmp_a_interval = self.a_interval.unsqueeze(-1) # shape: n_a,1,1
        for a in range(self.n_a):
            batch_similarities = [] # similarities, need to concatenate and calculate sum (equivalent to mean with argmax)
            for b_st in range(0, self.calib_size, self.calib_batch_size):
                b_ed = min(self.calib_size, b_st + self.calib_batch_size)
                x = self.raw_input[b_st:b_ed].cuda()
                raw_out_expanded = self.raw_out[b_st:b_ed].cuda().unsqueeze(-2) # shape: b,*,1,oc
                raw_grad = self.raw_grad[b_st:b_ed].cuda() # will be reshaped later
                similarities = []
                for p_st in range(0,self.eq_n,self.parallel_eq_n):
                    p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                    cur_a_interval = tmp_a_interval.repeat(1,1,p_ed-p_st) # shape: n_a,1,parallel_eq_n
                    cur_a_interval[a:a+1,:,:] = input_interval_candidates[a:a+1,:,p_st:p_ed]
                    # quantize weight and bias 
                    w_sim, bias_sim = self.quant_weight_bias()
                    # quantize input
                    x_sim=torch.cat(torch.chunk(x.unsqueeze(-2), chunks=self.n_a, dim=-1), dim=-2).unsqueeze(-1)
                    x_pos=(x_sim/(cur_a_interval)).round_().clamp_(0,self.a_qmax-1)*(cur_a_interval) # shape: b,*,n_a,crb_acts,parallel_eq_n
                    x_neg=(x_sim/(self.a_neg_interval)).round_().clamp_(-self.a_qmax,0)*(self.a_neg_interval) # shape: b,*,n_a,crb_acts,1
                    x_sim = (x_pos + x_neg).permute(*list(range(len(x_sim.shape)-3)),-1,-3,-2).reshape(*x.shape[:-1],p_ed-p_st,x.shape[-1]) # shape: b,*,parallel_eq_n,ic
                    # calculate similarity and store them
                    out_sim = F.linear(x_sim, w_sim, bias_sim) # shape: b,*,parallel_eq_n,oc
                    similarity = self._get_similarity(raw_out_expanded, out_sim, self.metric, raw_grad) # shape: b,*,parallel_eq_n
                    similarity = torch.mean(similarity, dim=list(range(1,len(similarity.shape)-1))) # shape: b, parallel_eq_n
                    similarity = torch.sum(similarity, dim=0, keepdim=True) # shape: 1, parallel_eq_n
                    similarities.append(similarity)
                # store best input interval and store in tmp_a_interval
                similarities = torch.cat(similarities, dim=1) # shape: 1, eq_n
                batch_similarities.append(similarities)
            batch_similarities = torch.cat(batch_similarities, dim=0).sum(dim=0, keepdim=False) # shape: eq_n
            a_best_index = batch_similarities.argmax(dim=0, keepdim=True).reshape(1,1,-1)
            tmp_a_interval[a:a+1,:,:] = torch.gather(input_interval_candidates[a:a+1,:,:],dim=2,index=a_best_index)
        self.a_interval = tmp_a_interval.squeeze(-1)