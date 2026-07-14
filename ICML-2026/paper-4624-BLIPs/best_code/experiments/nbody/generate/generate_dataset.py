from synthetic_sim import ChargedParticlesSim, SpringSim
import time
import numpy as np
import argparse

"""
nbody: python -u generate_dataset.py  --num-train 50000 --sample-freq 500 2>&1 | tee log_generating_100000.log &

nbody_small: python -u generate_dataset.py --num-train 10000 --seed 43 --sufix small 2>&1 | tee log_generating_10000_small.log &

"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--simulation", type=str, default="charged", help="What simulation to generate."
)
parser.add_argument(
    "--num-train",
    type=int,
    default=10000,
    help="Number of training simulations to generate.",
)
parser.add_argument(
    "--num-valid",
    type=int,
    default=2000,
    help="Number of validation simulations to generate.",
)
parser.add_argument(
    "--num-test", type=int, default=2000, help="Number of test simulations to generate."
)
parser.add_argument("--length", type=int, default=5000, help="Length of trajectory.")
parser.add_argument(
    "--length_test", type=int, default=5000, help="Length of test set trajectory."
)
parser.add_argument(
    "--sample-freq", type=int, default=100, help="How often to sample the trajectory."
)
parser.add_argument(
    "--n_balls", type=int, default=5, help="Number of balls in the simulation."
)
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--initial_vel", type=int, default=1, help="consider initial velocity"
)
parser.add_argument("--sufix", type=str, default="", help="add a sufix to the name")

args = parser.parse_args()

initial_vel_norm = 0.5
if not args.initial_vel:
    initial_vel_norm = 1e-16

if args.simulation == "springs":
    sim = SpringSim(noise_var=0.0, n_balls=args.n_balls)
    suffix = "_springs"
elif args.simulation == "charged":
    sim = ChargedParticlesSim(
        noise_var=0.0, n_balls=args.n_balls, vel_norm=initial_vel_norm
    )
    suffix = "_charged"
else:
    raise ValueError("Simulation {} not implemented".format(args.simulation))

suffix += str(args.n_balls) + "_initvel%d" % args.initial_vel + args.sufix
np.random.seed(args.seed)

print(suffix)


def generate_dataset(num_sims, length, sample_freq):
    loc_all = list()
    vel_all = list()
    edges_all = list()
    charges_all = list()
    for i in range(num_sims):
        t = time.time()
        loc, vel, edges, charges = sim.sample_trajectory(
            T=length, sample_freq=sample_freq
        )
        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))
        loc_all.append(loc)
        vel_all.append(vel)
        edges_all.append(edges)
        charges_all.append(charges)

    charges_all = np.stack(charges_all)
    loc_all = np.stack(loc_all)
    vel_all = np.stack(vel_all)
    edges_all = np.stack(edges_all)

    return loc_all, vel_all, edges_all, charges_all


if __name__ == "__main__":

    if args.num_train > 0:
        print("Generating {} training simulations".format(args.num_train))
        loc_train, vel_train, edges_train, charges_train = generate_dataset(
            args.num_train, args.length, args.sample_freq
        )
        np.save("loc_train" + suffix + ".npy", loc_train)
        np.save("vel_train" + suffix + ".npy", vel_train)
        np.save("edges_train" + suffix + ".npy", edges_train)
        np.save("charges_train" + suffix + ".npy", charges_train)

    if args.num_valid > 0:
        print("Generating {} validation simulations".format(args.num_valid))
        loc_valid, vel_valid, edges_valid, charges_valid = generate_dataset(
            args.num_valid, args.length, args.sample_freq
        )
        np.save("loc_valid" + suffix + ".npy", loc_valid)
        np.save("vel_valid" + suffix + ".npy", vel_valid)
        np.save("edges_valid" + suffix + ".npy", edges_valid)
        np.save("charges_valid" + suffix + ".npy", charges_valid)

    if args.num_test > 0:
        print("Generating {} test simulations".format(args.num_test))
        loc_test, vel_test, edges_test, charges_test = generate_dataset(
            args.num_test, args.length_test, args.sample_freq
        )
        np.save("loc_test" + suffix + ".npy", loc_test)
        np.save("vel_test" + suffix + ".npy", vel_test)
        np.save("edges_test" + suffix + ".npy", edges_test)
        np.save("charges_test" + suffix + ".npy", charges_test)
