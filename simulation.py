import time
import logging
import argparse
from stats import plot

OUTPUT_FOLDER = 'out/'
logging.basicConfig(level=logging.INFO)


def main(agent='basic'):
    logging.info("Starting simulation with agent - '" + agent + "'")
    if agent=='basic':
        results = run_simulation()
        plot.plot_results(results, OUTPUT_FOLDER + 'basic_agent_sim.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quadcopter reinforcement gym simulation')
    parser.add_argument('-a', '--agent', help='Simulation agent', required=True)
    args = vars(parser.parse_args())
    agent_name = args['agent']
    sim_start_time = time.time()
    main(agent_name)
    sim_end_time = time.time()
    logging.info("Simulation done in (sec): " + str(int(sim_end_time-sim_start_time)))
