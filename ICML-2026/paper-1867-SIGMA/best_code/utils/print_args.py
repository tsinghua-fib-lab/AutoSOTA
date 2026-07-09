def print_args(args):
    print("\033[1m" + "Basic Config" + "\033[0m")
    print(f'  {"Task Name:":<20}{args.task_name:<20}{"Is Training:":<20}{args.is_training:<20}')
    print(f'  {"Model ID:":<20}{args.model_id:<20}{"Model:":<20}{args.model:<20}')
    print()

    print("\033[1m" + "Data Loader" + "\033[0m")
    print(f'  {"Data:":<20}{args.data:<20}{"Root Path:":<20}{args.root_path:<20}')
    print(f'  {"Data Path:":<20}{args.data_path:<20}{"Features:":<20}{args.features:<20}')
    print(f'  {"Target:":<20}{args.target:<20}{"Freq:":<20}{args.freq:<20}')
    print(f'  {"Checkpoints:":<20}{args.checkpoints:<20}{"Save Checkpoints:":<20}{args.save_checkpoints:<20}')
    print()

    if args.task_name in ['long_term_forecast', 'short_term_forecast']:
        print("\033[1m" + "Forecasting Task" + "\033[0m")
        print(f'  {"Seq Len:":<20}{args.seq_len:<20}{"Label Len:":<20}{args.label_len:<20}')
        print(f'  {"Pred Len:":<20}{args.pred_len:<20}{"Seasonal Patterns:":<20}{args.seasonal_patterns:<20}')
        print(f'  {"Inverse:":<20}{args.inverse:<20}')
        print()

    print("\033[1m" + "Model Parameters" + "\033[0m")
    print(f'  {"Enc In:":<20}{args.enc_in:<20}{"Dec In:":<20}{args.dec_in:<20}')
    print(f'  {"C Out:":<20}{args.c_out:<20}{"d model:":<20}{args.d_model:<20}')
    print(f'  {"e layers:":<20}{args.e_layers:<20}{"d layers:":<20}{args.d_layers:<20}')
    print(f'  {"d FF:":<20}{args.d_ff:<20}{"Moving Avg:":<20}{args.moving_avg:<20}')
    print(f'  {"Dropout:":<20}{args.dropout:<20}{"Embed:":<20}{args.embed:<20}')
    print(f'  {"Activation:":<20}{args.activation:<20}{"Channel Independence:":<20}{args.channel_independence:<20}')
    print(f'  {"Use Norm:":<20}{args.use_norm:<20}')
    print()

    print("\033[1m" + "Run Parameters" + "\033[0m")
    print(f'  {"Num Workers:":<20}{args.num_workers:<20}{"Train Epochs:":<20}{args.train_epochs:<20}')
    print(f'  {"Batch Size:":<20}{args.batch_size:<20}{"Patience:":<20}{args.patience:<20}')
    print(f'  {"Learning Rate:":<20}{args.learning_rate:<20}{"Des:":<20}{args.des:<20}')
    print(f'  {"Loss:":<20}{args.loss:<20}{"Lradj:":<20}{args.lradj:<20}')
    print(f'  {"Use Amp:":<20}{args.use_amp:<20}{"Seed:":<20}{args.seed:<20}')
    print()

    print("\033[1m" + "GPU" + "\033[0m")
    print(f'  {"Use GPU:":<20}{args.use_gpu:<20}{"GPU:":<20}{args.gpu:<20}')
    print(f'  {"GPU Type:":<20}{args.gpu_type:<20}{"Use Multi GPU:":<20}{args.use_multi_gpu:<20}')
    print(f'  {"Devices:":<20}{args.devices:<20}')
    print()

    print("\033[1m" + "SiGMA Specific" + "\033[0m")
    print(f'  {"Scale Independence:":<20}{args.scale_independence:<20}{"Feature Transformation:":<20}{args.feature_transformation:<20}')
    print()

    if hasattr(args, 'use_wandb') and args.use_wandb:
        print("\033[1m" + "Wandb Config" + "\033[0m")
        print(f'  {"Use Wandb:":<20}{args.use_wandb:<20}{"Wandb Project:":<20}{args.wandb_project:<20}')
        if hasattr(args, 'wandb_entity') and args.wandb_entity:
            print(f'  {"Wandb Entity:":<20}{args.wandb_entity:<20}')
        if hasattr(args, 'wandb_run_name') and args.wandb_run_name:
            print(f'  {"Wandb Run Name:":<20}{args.wandb_run_name:<20}')
        print()
