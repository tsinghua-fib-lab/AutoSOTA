load MTOData_Temp.mat

tasks = 18;
reps = 10;
gen = 500;

MTOData.Algorithms = repmat(MTOData.Algorithms(1), 1, 2);
MTOData.Algorithms(1).Name = 'PPO';
MTOData.Algorithms(2).Name = 'A2C';
MTOData.Algorithms(1).Para = {'HiddenSize', num2str(64)};
MTOData.Algorithms(2).Para = {'HiddenSize', num2str(64)};

MTOData.Results = [];
for algo = 1:2
    switch algo
        case 1
            load ppo_results.mat
        case 2
            load a2c_results.mat
    end
    for rep = 1:reps
        for task = 1:tasks
            ppodata = gen2eva_rl(squeeze(all_tasks(task, rep, :))', 1:size(all_tasks, 3), gen);
            MTOData.Results(1, algo, rep).Obj(task, 1:gen) = -ppodata;
            MTOData.Results(1, algo, rep).CV(task, 1:gen) = zeros(1, gen);
        end
    end
end

MTOData.RunTimes = zeros(1, 2, reps);

save('RL-Compare', 'MTOData');
