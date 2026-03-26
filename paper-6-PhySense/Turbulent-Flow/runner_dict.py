from runners import exp_senseiver_rect_flow_pipe_walk, exp_senseiver_rect_flow_pipe

def get_runner(args, config):
    model_dict = {
        'exp_senseiver_rect_flow_pipe': exp_senseiver_rect_flow_pipe,
        'exp_senseiver_rect_flow_pipe_walk': exp_senseiver_rect_flow_pipe_walk,
    }
    return model_dict[args.runner].Diffusion(args, config)