import csv
import time
import logging
import numpy as np
from task import Task
from agents.basic_agent import Basic_Agent
from stats import plot


OUTPUT_FOLDER = 'out/'
logging.basicConfig(level=logging.INFO)


def run_simulation(output_folder=OUTPUT_FOLDER):
    # Modify the values below to give the quadcopter a different starting position.
    runtime = 5.                                     # time limit of the episode
    init_pose = np.array([0., 0., 10., 0., 0., 0.])  # initial pose
    init_velocities = np.array([0., 0., 0.])         # initial velocities
    init_angle_velocities = np.array([0., 0., 0.])   # initial angle velocities
    file_output = output_folder + 'data.txt'         # file name for saved results

    # Setup
    task = Task(init_pose, init_velocities, init_angle_velocities, runtime)
    agent = Basic_Agent(task)
    done = False
    labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
              'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
              'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4']
    results = {x : [] for x in labels}

    # Run the simulation, and save the results.
    with open(file_output, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(labels)
        while True:
            rotor_speeds = agent.act()
            _, _, done = task.step(rotor_speeds)
            to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(rotor_speeds)
            for ii in range(len(labels)):
                results[labels[ii]].append(to_write[ii])
            writer.writerow(to_write)
            if done:
                break
    return results


def main():
    results = run_simulation()
    plot.plot_results(results, OUTPUT_FOLDER + 'basic_agent_sim.png')


if __name__ == "__main__":
    logging.info("Starting simulation")
    sim_start_time = time.time()
    main()
    sim_end_time = time.time()
    logging.info("Simulation done in (sec): " + str(int(sim_end_time-sim_start_time)))
